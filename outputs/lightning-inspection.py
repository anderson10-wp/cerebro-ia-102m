import json
from pathlib import Path
from lightning_sdk import User
from lightning_sdk.api.utils import cached_lightning_client

client = cached_lightning_client()
account = client.auth_service_get_user()
report = {'authenticated': True, 'username': account.username, 'organizations': [], 'teamspaces': [], 'read_only': True}
for org in account.organizations or []:
    item = {'name': org.name, 'id': org.id}
    try:
        item['balance'] = client.billing_service_get_org_balance(org.id).to_dict()
    except Exception as error:
        item['balance_error'] = type(error).__name__
    report['organizations'].append(item)
user = User(account.username)
for team in user.teamspaces:
    item = {'name': team.name, 'id': team.id}
    try:
        item['balance'] = client.billing_service_get_project_balance(team.id).to_dict()
    except Exception as error:
        item['balance_error'] = type(error).__name__
    try:
        item['studios'] = [{'name': s.name, 'status': str(s.status)} for s in team.studios]
    except Exception as error:
        item['studios_error'] = type(error).__name__
    try:
        machines = team.list_machines()
        item['machines'] = [{'name': m.name, 'cost': m.cost, 'provider': str(m.provider)} for m in machines]
    except Exception as error:
        item['machines_error'] = type(error).__name__
    report['teamspaces'].append(item)
Path(__file__).with_suffix('.json').write_text(json.dumps(report, default=str, indent=2), encoding='utf-8')
print(json.dumps(report, default=str))
