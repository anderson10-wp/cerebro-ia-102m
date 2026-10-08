from pathlib import Path
import hashlib, json, sys, zipfile, shutil
root=Path(__file__).resolve().parent
assets=Path(sys.argv[1] if len(sys.argv)>1 else 'arquivos-release').resolve()
manifest=json.loads((root/'archive-manifest.json').read_text(encoding='utf-8'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
for obj in manifest['objects']:
 archive=assets/obj['asset']
 if sha(archive)!=obj['asset_sha256']:raise SystemExit('Asset corrompido: '+archive.name)
 temp=assets/(obj['sha256']+'.restoring')
 with zipfile.ZipFile(archive) as z, z.open('payload') as src, temp.open('wb') as dst:shutil.copyfileobj(src,dst)
 if sha(temp)!=obj['sha256']:raise SystemExit('Conteúdo corrompido: '+archive.name)
 for name in obj['paths']:
  target=(root/name).resolve()
  if not target.is_relative_to(root):raise SystemExit('Caminho inválido')
  if target.exists():
   if sha(target)!=obj['sha256']:raise SystemExit('Arquivo diferente já existe: '+name)
   continue
  target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(temp,target)
 temp.unlink();print('Restaurado:',len(obj['paths']),'caminho(s)')
print('Restauração completa.')
