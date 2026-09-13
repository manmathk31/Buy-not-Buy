import sys
from pathlib import Path

code_dir = Path(__file__).resolve().parent.parent / "code"
sys.path.insert(0, str(code_dir))

from extraction import load_env_and_detect_key
load_env_and_detect_key(verbose=True)

from loaders import load_dataset
from extraction import ExtractionLayer, apply_extractions_to_events

dataset_dir = Path(__file__).resolve().parent.parent / "dataset"
store = load_dataset(dataset_dir)

# PART 1: Check request_03 (event_253, image_01) and request_19 (event_1700, image_04)
extractor = ExtractionLayer()
image_results = {img.image_id: extractor.extract_image_amount(img, dataset_dir, i + 1, len(store.images)) for i, img in enumerate(store.images)}
message_amendments = [extractor.parse_message(m) for m in store.messages]
corrected_events = apply_extractions_to_events(store.events, image_results, message_amendments)

raw_event_map = {e.event_id: e for e in store.events}
corr_event_map = {e.event_id: e for e in corrected_events}

print("\n--- PART 1 VERIFICATION ---")
for req_id, ev_id, img_id in [("request_03", "event_253", "image_01"), ("request_19", "event_1700", "image_04")]:
    raw_ev = raw_event_map.get(ev_id)
    corr_ev = corr_event_map.get(ev_id)
    img_res = image_results.get(img_id)
    print(f"Request: {req_id} | Event: {ev_id} | Image: {img_id}")
    print(f"  Raw event amount: {raw_ev.amount if raw_ev else None}")
    print(f"  Extracted image amount from API: {img_res.extracted_amount if img_res else None}")
    print(f"  Corrected event amount: {corr_ev.amount if corr_ev else None}")
    print(f"  Event direction: {corr_ev.direction if corr_ev else None}, status: {corr_ev.status if corr_ev else None}, date: {corr_ev.event_date if corr_ev else None}")
