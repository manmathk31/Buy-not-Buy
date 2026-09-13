import sys
from pathlib import Path

code_dir = Path(__file__).resolve().parent.parent / "code"
sys.path.insert(0, str(code_dir))

from extraction import load_env_and_detect_key
load_env_and_detect_key(verbose=True)

from loaders import load_dataset
from extraction import ExtractionLayer, apply_extractions_to_events
from forecast import ForecastEngine
from decision import DecisionEngine

dataset_dir = Path(__file__).resolve().parent.parent / "dataset"
store = load_dataset(dataset_dir)
forecast_engine = ForecastEngine(store.currency_converter)

# PART 1: Request 03 and Request 19 trace
print("\n==========================================================================")
print("PART 1 DIAGNOSTIC: Image Extraction Trace for Request 03 and Request 19")
print("==========================================================================")

extractor = ExtractionLayer()
image_results = {img.image_id: extractor.extract_image_amount(img, dataset_dir, i + 1, len(store.images)) for i, img in enumerate(store.images)}
message_amendments = [extractor.parse_message(m) for m in store.messages]
corrected_events = apply_extractions_to_events(store.events, image_results, message_amendments)

events_by_user_raw = {}
for e in store.events:
    events_by_user_raw.setdefault(e.user_id, []).append(e)

events_by_user_corr = {}
for e in corrected_events:
    events_by_user_corr.setdefault(e.user_id, []).append(e)

for req_id, ev_id, img_id in [("request_03", "event_253", "image_01"), ("request_19", "event_1700", "image_04")]:
    req = store.get_context_for_request(req_id).request
    prof = store.get_context_for_request(req_id).profile
    
    raw_evs = events_by_user_raw.get(req.user_id, [])
    corr_evs = events_by_user_corr.get(req.user_id, [])
    
    # Run forecast without extraction (raw events) vs with extraction (corrected events)
    res_raw = forecast_engine.run_forecast(req, prof, raw_evs)
    res_corr = forecast_engine.run_forecast(req, prof, corr_evs)
    
    raw_ev_obj = next((e for e in raw_evs if e.event_id == ev_id), None)
    corr_ev_obj = next((e for e in corr_evs if e.event_id == ev_id), None)
    
    print(f"\nRequest: {req_id} (User: {req.user_id})")
    print(f"  Target Event ID: {ev_id} | Linked Image ID: {img_id}")
    print(f"  - Amount BEFORE extraction (raw event CSV): {raw_ev_obj.amount if raw_ev_obj else None}")
    print(f"  - Amount extracted by Gemini API from image: {image_results[img_id].extracted_amount}")
    print(f"  - Amount AFTER extraction (flowing to forecast): {corr_ev_obj.amount if corr_ev_obj else None}")
    print(f"  - Target event details: status='{corr_ev_obj.status}', direction='{corr_ev_obj.direction}', event_date='{corr_ev_obj.event_date}', settlement_date='{corr_ev_obj.settlement_date}'")
    print(f"  - Forecast amount_safe_to_pay BEFORE extraction: {res_raw.amount_safe_to_pay:.2f}")
    print(f"  - Forecast amount_safe_to_pay AFTER extraction:  {res_corr.amount_safe_to_pay:.2f}")
    print(f"  - Min margin date: {min(res_corr.daily_margins, key=res_corr.daily_margins.get)}")

# PART 2: Lookback windows & drag double-counting for Request 08, 14, 22
print("\n==========================================================================")
print("PART 2 DIAGNOSTIC: Lookback Windows & Drag Tracing for Requests 08, 14, 22")
print("==========================================================================")

for req_id in ["request_08", "request_14", "request_15", "request_21", "request_22"]:
    ctx = store.get_context_for_request(req_id)
    req = ctx.request
    prof = ctx.profile
    u_events = events_by_user_corr.get(req.user_id, [])
    
    req_d = forecast_engine.parse_date(req.request_date)
    
    # Run forecast to see detailed result
    f_res = forecast_engine.run_forecast(req, prof, u_events)
    
    # Ground truth
    sample_rec = next((s for s in store.sample_requests if s.request_id == req_id), None)
    true_safe = sample_rec.amount_safe_to_pay if sample_rec else 0.0
    
    print(f"\n--- {req_id} (User: {req.user_id}, Req Date: {req.request_date}) ---")
    print(f"  Requested Amt: {req.requested_amount:.2f} | Safe Amt Pred: {f_res.amount_safe_to_pay:.2f} | True: {true_safe:.2f} | Diff: {abs(f_res.amount_safe_to_pay - true_safe):.2f}")
    print(f"  Protected categories: {prof.protect_categories_list}")
    print(f"  Current Balance: {prof.current_available_balance:.2f} | Min Balance to Keep: {prof.minimum_balance_to_keep:.2f}")
    
    # Inspect recurrence detector vs baseline drag lookback
    print(f"  Recurring groups detected ({len(f_res.recurring_groups_detected)}):")
    for rg in f_res.recurring_groups_detected:
        print(f"    - Cat: {rg['category']}, Desc: {rg['description']}, Amount: {rg['amount']}, Interval: {rg['median_interval']}, LastDate: {rg['last_date']}")
        
    # Check historical events for user in protected categories
    print("  Historical events in protected categories:")
    for e in u_events:
        cat = e.category.strip().lower()
        e_date = forecast_engine.parse_date(e.event_date)
        s_date = forecast_engine.parse_date(e.settlement_date if e.settlement_date else e.event_date)
        if cat in [c.strip().lower() for c in prof.protect_categories_list]:
            days_ago = (req_d - e_date).days
            print(f"    - EvID: {e.event_id}, Cat: {cat}, Amt: {e.amount}, Status: {e.status}, Direction: {e.direction}, EvDate: {e.event_date} ({days_ago}d ago), SettledDate: {e.settlement_date}")
            
    # Print day 0 (request_date) cash flows and margins
    print(f"  Day 0 ({req.request_date}) balance: {f_res.daily_balances[req_d]:.2f}, margin: {f_res.daily_margins[req_d]:.2f}")
