"""Launch once and retain the server's actual error message without credentials."""
import json,sys
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
api=KaggleApi(); api.authenticate()
try:
    folder=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
    timeout=int(sys.argv[2]) if len(sys.argv)>2 else 5400
    response=api.kernels_push(str(folder),timeout=timeout)
    print(response)
except Exception as exc:
    response=getattr(exc,'response',None)
    message=response.text[:3000] if response is not None else type(exc).__name__
    print(json.dumps({'status':getattr(response,'status_code',None),'message':message}))
    raise SystemExit(1)
