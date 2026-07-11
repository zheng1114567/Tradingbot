"""Standalone auditable data agent.

The agent records each stage of a data run:
1. input request
2. raw vendor data
3. cleaned data
4. analysis data
5. final layered response
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from ..config import config
from .cleaner import DataCleaner
from .factors import FactorCalculator
from .manifest import DataManifest
from .vendor_router import get_vendor_chain, route_to_vendor


RouteFn = Callable[..., Any]


_SAFE_PATH_PART = re.compile(r"[^A-Za-z0-9_.-]+")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_path_part(value: str, fallback: str) -> str:
    safe = _SAFE_PATH_PART.sub("_", value).strip("._")
    return safe or fallback


def _records_from_frame(df: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
    frame = df if limit is None else df.tail(limit)
    return json.loads(frame.to_json(orient="records", force_ascii=False, date_format="iso"))


@dataclass(frozen=True)
class DataAgentRequest:
    """Input boundary for a standalone data-agent run."""

    ticker: str
    trade_date: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    include_capital_flow: bool = True
    include_factors: bool = True
    output_dir: str | None = None
    max_return_records: int = 20

    def normalized_trade_date(self) -> str:
        return self.trade_date or date.today().isoformat()

    def normalized_end_date(self) -> str | None:
        if self.end_date:
            return self.end_date
        if self.trade_date:
            return self.trade_date.replace("-", "")
        return None


@dataclass
class DataAgentArtifact:
    """One persisted step in the data-agent trace."""

    stage: str
    path: str
    record_count: int | None = None
    columns: list[str] = field(default_factory=list)


@dataclass
class DataAgentRun:
    """Structured return payload for a standalone data-agent run."""

    run_id: str
    request: dict[str, Any]
    artifacts: dict[str, DataAgentArtifact]
    manifest_path: str
    response_path: str
    final_data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["artifacts"] = {k: asdict(v) for k, v in self.artifacts.items()}
        return payload


class DataAgent:
    """Collect, clean, analyze, and persist market data in auditable layers."""

    def __init__(
        self,
        *,
        route_fn: RouteFn = route_to_vendor,
        results_dir: str | None = None,
    ) -> None:
        self._route_fn = route_fn
        self._results_dir = Path(results_dir or config.get("results_dir", "data/results"))

    def run(self, request: DataAgentRequest) -> DataAgentRun:
        run_dir = self._make_run_dir(request)
        manifest = DataManifest(
            ticker=request.ticker,
            trade_date=request.normalized_trade_date(),
        )

        artifacts: dict[str, DataAgentArtifact] = {}
        input_payload = {
            "stage": "input",
            "created_at": _utc_now(),
            "request": asdict(request),
            "vendor_chain": {
                "daily": get_vendor_chain("get_daily"),
                "capital_flow": get_vendor_chain("get_capital_flow"),
                "factors": get_vendor_chain("get_factors"),
            },
        }
        artifacts["input"] = self._write_json(run_dir / "01_input" / "request.json", input_payload)

        raw_payload = self._collect_raw(request, manifest)
        artifacts["raw"] = self._write_json(run_dir / "02_raw" / "raw_data.json", raw_payload)

        cleaned_payload = self._clean(raw_payload)
        artifacts["cleaned"] = self._write_json(run_dir / "03_cleaned" / "cleaned_data.json", cleaned_payload)

        analysis_payload = self._analyze(cleaned_payload, request)
        artifacts["analysis"] = self._write_json(
            run_dir / "04_analysis" / "analysis_data.json",
            analysis_payload,
        )

        final_payload = {
            "stage": "final",
            "created_at": _utc_now(),
            "input": input_payload,
            "raw": raw_payload,
            "cleaned": cleaned_payload,
            "analysis": analysis_payload,
            "manifest": manifest.to_dict(),
        }
        artifacts["final"] = self._write_json(run_dir / "05_final" / "response.json", final_payload)

        manifest_path = manifest.save(results_dir=str(run_dir / "05_final"))

        return DataAgentRun(
            run_id=run_dir.name,
            request=asdict(request),
            artifacts=artifacts,
            manifest_path=str(manifest_path),
            response_path=artifacts["final"].path,
            final_data=final_payload,
        )

    def _make_run_dir(self, request: DataAgentRequest) -> Path:
        ticker = _safe_path_part(request.ticker.replace(".", "_"), "unknown_ticker")
        trade_date = _safe_path_part(request.normalized_trade_date(), "unknown_date")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self._results_dir / "data_agent_runs" / f"{trade_date}_{ticker}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _collect_raw(self, request: DataAgentRequest, manifest: DataManifest) -> dict[str, Any]:
        daily = self._safe_route(
            "get_daily",
            manifest,
            field_name="stock.daily",
            code=request.ticker,
            start_date=request.start_date,
            end_date=request.normalized_end_date(),
        )
        capital_flow = []
        if request.include_capital_flow:
            capital_flow = self._safe_route(
                "get_capital_flow",
                manifest,
                field_name="stock.capital_flow",
                code=request.ticker,
                start_date=request.start_date,
                end_date=request.normalized_end_date(),
            )

        return {
            "stage": "raw",
            "created_at": _utc_now(),
            "daily": daily,
            "capital_flow": capital_flow,
        }

    def _safe_route(
        self,
        method: str,
        manifest: DataManifest,
        *,
        field_name: str,
        **kwargs: Any,
    ) -> Any:
        vendor_chain = get_vendor_chain(method)
        try:
            result = self._route_fn(method, **kwargs)
            is_no_data = isinstance(result, str) and result.startswith("NO_DATA_AVAILABLE")
            count = len(result) if isinstance(result, list) else None
            manifest.add_field(
                field_name,
                available=not is_no_data,
                source=f"vendor_router:{method}",
                vendor_chain=vendor_chain,
                fallback_used=len(vendor_chain) > 1,
                error=result if is_no_data else None,
                record_count=count,
            )
            return result
        except Exception as exc:
            manifest.add_field(
                field_name,
                available=False,
                source=f"vendor_router:{method}",
                vendor_chain=vendor_chain,
                error=str(exc),
            )
            return {
                "error": str(exc),
                "method": method,
                "vendor_chain": vendor_chain,
            }

    def _clean(self, raw_payload: dict[str, Any]) -> dict[str, Any]:
        daily_raw = raw_payload.get("daily")
        if isinstance(daily_raw, list):
            daily_df = DataCleaner.clean_daily(daily_raw)
            daily_df = DataCleaner.detect_limit_up_down(daily_df)
        else:
            daily_df = pd.DataFrame()

        capital_flow_raw = raw_payload.get("capital_flow")
        capital_flow = capital_flow_raw if isinstance(capital_flow_raw, list) else []

        return {
            "stage": "cleaned",
            "created_at": _utc_now(),
            "daily": {
                "record_count": int(len(daily_df)),
                "columns": list(daily_df.columns),
                "records": _records_from_frame(daily_df),
            },
            "capital_flow": {
                "record_count": len(capital_flow),
                "records": capital_flow,
            },
        }

    def _analyze(self, cleaned_payload: dict[str, Any], request: DataAgentRequest) -> dict[str, Any]:
        daily_records = cleaned_payload.get("daily", {}).get("records", [])
        daily_df = pd.DataFrame(daily_records)
        factor_records: list[dict[str, Any]] = []
        summary: dict[str, Any] = {
            "ticker": request.ticker,
            "record_count": int(len(daily_df)),
            "latest": None,
            "price_change_pct": None,
            "volume_latest": None,
            "amount_latest": None,
        }

        if not daily_df.empty:
            if "trade_date" in daily_df.columns:
                daily_df["trade_date"] = pd.to_datetime(daily_df["trade_date"])
                daily_df = daily_df.sort_values("trade_date")
            latest = daily_df.iloc[-1].to_dict()
            summary.update(
                {
                    "latest": self._json_safe_record(latest),
                    "price_change_pct": self._json_safe_value(latest.get("pct_chg")),
                    "volume_latest": self._json_safe_value(latest.get("volume")),
                    "amount_latest": self._json_safe_value(latest.get("amount")),
                }
            )

        if request.include_factors and not daily_df.empty:
            factor_df = FactorCalculator.run_all(daily_df.copy())
            factor_records = _records_from_frame(factor_df, limit=request.max_return_records)

        return {
            "stage": "analysis",
            "created_at": _utc_now(),
            "summary": summary,
            "factors": {
                "record_count": len(factor_records),
                "records": factor_records,
            },
        }

    def _write_json(self, path: Path, payload: dict[str, Any]) -> DataAgentArtifact:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return DataAgentArtifact(
            stage=str(payload.get("stage", path.parent.name)),
            path=str(path),
            record_count=self._record_count(payload),
            columns=self._columns(payload),
        )

    @staticmethod
    def _record_count(payload: dict[str, Any]) -> int | None:
        if isinstance(payload.get("daily"), list):
            return len(payload["daily"])
        daily = payload.get("daily")
        if isinstance(daily, dict) and "record_count" in daily:
            return int(daily["record_count"])
        return None

    @staticmethod
    def _columns(payload: dict[str, Any]) -> list[str]:
        daily = payload.get("daily")
        if isinstance(daily, dict):
            return list(daily.get("columns", []))
        return []

    @staticmethod
    def _json_safe_value(value: Any) -> Any:
        if pd.isna(value):
            return None
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return value

    @classmethod
    def _json_safe_record(cls, record: dict[str, Any]) -> dict[str, Any]:
        return {key: cls._json_safe_value(value) for key, value in record.items()}


def run_data_agent(
    ticker: str,
    *,
    trade_date: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    output_dir: str | None = None,
) -> DataAgentRun:
    """Convenience entry point for tests and CLI usage."""

    request = DataAgentRequest(
        ticker=ticker,
        trade_date=trade_date,
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir,
    )
    return DataAgent(results_dir=output_dir).run(request)
