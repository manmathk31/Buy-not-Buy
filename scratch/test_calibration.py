import sys
from pathlib import Path

code_dir = Path(__file__).resolve().parent.parent / "code"
sys.path.insert(0, str(code_dir))

from loaders import load_dataset
from extraction import ExtractionLayer, apply_extractions_to_events
from forecast import ForecastEngine
from models import Request

dataset_dir = Path(__file__).resolve().parent.parent / "dataset"
store = load_dataset(dataset_dir)
extractor = ExtractionLayer()
image_results = {img.image_id: extractor.extract_image_amount(img, dataset_dir, i + 1, len(store.images)) for i, img in enumerate(store.images)}
message_amendments = [extractor.parse_message(m) for m in store.messages]
corrected_events = apply_extractions_to_events(store.events, image_results, message_amendments)

events_by_user = {}
for e in corrected_events:
    events_by_user.setdefault(e.user_id, []).append(e)

forecast_engine = ForecastEngine(store.currency_converter)

print("\n--- TESTING 90-DAY DRAG WINDOW CALIBRATION ---")
for sample in store.sample_requests:
    req = store.get_context_for_request(sample.request_id).request
    prof = store.get_context_for_request(sample.request_id).profile
    u_events = events_by_user.get(req.user_id, [])
    
    f_res = forecast_engine.run_forecast(req, prof, u_events)
    diff = abs(f_res.amount_safe_to_pay - sample.amount_safe_to_pay)
    
    if diff > 0.01:
        print(f"{sample.request_id:<11} | Pred: {f_res.amount_safe_to_pay:>10.2f} | True: {sample.amount_safe_to_pay:>10.2f} | Diff: {diff:>9.2f}")
