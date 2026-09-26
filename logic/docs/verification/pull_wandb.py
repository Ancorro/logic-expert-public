import wandb, json
api = wandb.Api(timeout=60)
ent = "ancorro-oregon-state-university"
projs = [p.name for p in api.projects(ent)]
print("projects:", projs)
rows=[]
for p in projs:
    for r in api.runs(f"{ent}/{p}"):
        s=r.summary._json_dict; c=r.config
        m=c.get("model",{}) if isinstance(c.get("model"),dict) else {}
        t=c.get("train",{}) if isinstance(c.get("train"),dict) else {}
        d=c.get("data",{}) if isinstance(c.get("data"),dict) else {}
        rows.append(dict(project=p,id=r.id,name=r.name,group=r.group,state=r.state,created=r.created_at,
          variant=m.get("variant"),axis=m.get("capacity_window_axis"),G=m.get("num_gates"),alpha=m.get("alpha_init"),
          fusion=m.get("fusion_mode"),nogate=m.get("use_no_gate_stream"),learn_alpha=m.get("learn_alpha", m.get("fusion_alpha_learnable")),
          seed=t.get("seed"),epochs=t.get("epochs"),task=d.get("task", d.get("dataset")),
          acc=s.get("val/acc"),loss=s.get("val/loss"),acc_min=s.get("val/acc_min"),fa=s.get("train/fusion_alpha"),
          ent=s.get("train/routing_entropy"),gn=s.get("train/grad_norm"),params=s.get("perf/trainable_params"),
          skeys=[k for k in s if not k.startswith("_")][:200]))
json.dump(rows,open("wandb_runs.json","w"),default=str,indent=1)
print(len(rows))
