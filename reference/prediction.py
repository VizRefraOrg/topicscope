from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from starlette.requests import Request

from core import security
from models.payload import EntityPayload
from models.prediction import EntityResult, RefineEntityResult, RawEntityResult, DistanceMatrixResult, PlotlyGraphModel
from services.graph.helper import PlotlyGraphClient
from services.models import LanguageModel

router = APIRouter()


@router.post("/distance", response_model=DistanceMatrixResult, response_description="Entity extraction")
def get_distance(
        request: Request,
        authenticated: bool = Depends(security.validate_request),
        block_data: EntityPayload = None
) -> DistanceMatrixResult:
    model: LanguageModel = request.app.state.model

    clean_text, uuid = model.pre_process(block_data)
    if len(clean_text.split()) < 20:
        raise HTTPException(status_code=400, detail=f"Invalid text content: too few tokens (words) to process.")

    prediction: EntityResult = model.predict(block_data)
    response: DistanceMatrixResult = prediction.distances

    return response


@router.post("/refine", response_model=RefineEntityResult, response_description="Entity extraction")
def get_circles(
        request: Request,
        authenticated: bool = Depends(security.validate_request),
        block_data: EntityPayload = None
) -> RefineEntityResult:
    model: LanguageModel = request.app.state.model

    clean_text, uuid = model.pre_process(block_data)
    if len(clean_text.split()) < 20:
        raise HTTPException(status_code=400, detail=f"Invalid text content: too few tokens (words) to process.")

    prediction: EntityResult = model.predict(block_data)
    response: RefineEntityResult = prediction.circles
    return response


@router.post("/raw", response_model=RawEntityResult, response_description="Entity extraction")
def get_raw(
        request: Request,
        authenticated: bool = Depends(security.validate_request),
        block_data: EntityPayload = None
) -> RawEntityResult:
    model: LanguageModel = request.app.state.model

    clean_text, uuid = model.pre_process(block_data)
    if len(clean_text.split()) < 20:
        raise HTTPException(status_code=400, detail=f"Invalid text content: too few tokens (words) to process.")

    prediction: EntityResult = model.predict(block_data)
    response: RawEntityResult = prediction.raw

    return response


@router.post("/graph", response_model=PlotlyGraphModel, response_description="Generate 3D graph")
def graph_3d(
        request: Request,
        authenticated: bool = Depends(security.validate_request),
        text_id: Optional[str] = None,
        block_data: Optional[EntityPayload] = None
) -> RawEntityResult:
    model: LanguageModel = request.app.state.model
    graph_client: PlotlyGraphClient = request.app.state.graph_client

    clean_text, uuid = model.pre_process(block_data)
    if clean_text is None and text_id is None:
        raise HTTPException(status_code=400, detail=f"Must include text query in payload or text_id in query.")
    elif clean_text is not None and len(clean_text.split()) < 20:
        raise HTTPException(status_code=400, detail=f"Invalid text content: too few tokens (words) to process.")

    prediction: EntityResult = model.predict(block_data, text_id)
    data: pd.DataFrame = pd.json_normalize(prediction.circles.entities)
    graph: PlotlyGraphModel = graph_client.fit(data).create_3d_graph().persist(bucket="vizrefra-public")
    return graph


@router.post("/quick_reading", response_model=PlotlyGraphModel, response_description="Generate quick reading graph")
def quick_reading(
        request: Request,
        block_data: Optional[EntityPayload],
        authenticated: bool = Depends(security.validate_request),
) -> RawEntityResult:
    model: LanguageModel = request.app.state.model
    graph_client: PlotlyGraphClient = request.app.state.graph_client

    clean_text, uuid = model.pre_process(block_data)
    if clean_text is None:
        raise HTTPException(status_code=400, detail=f"Must send an input text.")
    elif clean_text is not None and len(clean_text.split()) < 20:
        raise HTTPException(status_code=400, detail=f"Invalid text content: too few tokens (words) to process.")

    prediction: EntityResult = model.predict(block_data)
    # prepare data
    entities = pd.json_normalize(prediction.raw.entities)
    entities['offset'] = entities['entity'].apply(lambda x: clean_text.find(x))
    merged_df = entities[entities['offset'] > -1].sort_values('offset').reset_index(drop=True)
    merged_df['concat_entity'] = merged_df.apply(
        lambda row: f"{row['entity']} ({row['tag']})" if row['tag'] != 'OTHER' else row['entity'], axis=1)

    graph: PlotlyGraphModel = graph_client.fit(merged_df).create_quick_reading().persist(bucket="vizrefra-public")
    return graph
