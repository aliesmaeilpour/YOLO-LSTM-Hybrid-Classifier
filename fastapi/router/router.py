from fastapi import APIRouter, File, UploadFile, HTTPException, Request, Depends
from schemas import Classification
from classification.yololstmhybrid import Classifier

router = APIRouter(
    prefix='/ocr',
    tags=['PriceDateBarcodeStoreNameOCR']
)

def get_model(request: Request) -> Classifier:
    return request.app.state.model

@router.post('/', response_model=Classification)
async def get_uploadfile(upload_file: UploadFile=File(...),
                         model: Classifier = Depends(get_model)):

    contents = await upload_file.read()
    try:
        # yt = Yolov8TrOCR(chunked=contents)
        classes = await model(contents)

        return Classification(
            classes=classes
        )
    
    except:
        HTTPException(status_code=500, detail="Image cannot be processed.")