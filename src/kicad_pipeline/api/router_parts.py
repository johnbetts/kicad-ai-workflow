"""Parts search endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from kicad_pipeline.api.schemas import PartSearchResponseSchema, PartSearchResultSchema

router = APIRouter()


@router.get("/search", response_model=PartSearchResponseSchema)
async def search_parts(
    q: str,
    basic_only: bool = False,
    in_stock: bool = False,
    limit: int = 20,
) -> PartSearchResponseSchema:
    """Search JLCPCB parts database."""
    try:
        from kicad_pipeline.parts.jlcpcb_db import JLCPCBPartsDB

        with JLCPCBPartsDB() as db:
            parts = db.search(q, limit=limit * 2)  # Over-fetch for filtering

            results: list[PartSearchResultSchema] = []
            for p in parts:
                if basic_only and not p.basic:
                    continue
                if in_stock and p.stock <= 0:
                    continue
                results.append(
                    PartSearchResultSchema(
                        lcsc=p.lcsc,
                        mfr=p.mfr or "",
                        description=p.description or "",
                        package=p.package or "",
                        stock=p.stock,
                        price=p.price,
                        basic=p.basic,
                    )
                )
                if len(results) >= limit:
                    break

            return PartSearchResponseSchema(
                query=q,
                results=results,
                total=len(results),
            )
    except Exception:
        # JLCPCB DB not available — return empty results
        return PartSearchResponseSchema(query=q, results=[], total=0)
