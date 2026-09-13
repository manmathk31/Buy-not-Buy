import sys
from pathlib import Path

code_dir = Path(__file__).resolve().parent.parent / "code"
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from loaders import load_dataset
from forecast import ForecastEngine
from models import Request

def diag_16():
    dataset_dir = code_dir.parent / "dataset"
    store = load_dataset(dataset_dir)
    engine = ForecastEngine(store.currency_converter)

    req_id = "request_16"
    s = [x for x in store.sample_requests if x.request_id == req_id][0]
    ctx = store.get_context_for_request(req_id)
    req = Request(s.request_id, s.user_id, s.request_date, s.request_type, s.requested_amount, s.desired_completion_date, s.allows_partial_payment, s.request_text)

    res = engine.run_forecast(req, ctx.profile, ctx.user_events)
    print(f"=== DIAGNOSTIC FOR {req_id} (User {s.user_id}) ===")
    print(f"Profile available balance: {ctx.profile.current_available_balance} {ctx.profile.home_currency}")
    print(f"Minimum balance to keep: {ctx.profile.minimum_balance_to_keep}")
    print(f"Protect categories: {ctx.profile.expense_categories_to_protect}")
    print(f"Calculated Safe Amount: {res.amount_safe_to_pay}")
    print(f"True Safe Amount: {s.amount_safe_to_pay}")
    print(f"Calculated Min Margin: {res.min_margin}")
    print("Daily margins around min:")
    for d, m in sorted(res.daily_margins.items()):
        if m < 130000:
            print(f"  {d}: bal={res.daily_balances[d]}, margin={m}")

if __name__ == "__main__":
    diag_16()
