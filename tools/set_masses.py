"""질량 보정 도구.
사용법
  python set_masses.py --total 32.5              # 전체 질량에 맞춰 모든 링크를 같은 비율로 스케일
  python set_masses.py --csv link_masses.csv     # cad_mass_kg 열에 값이 있는 링크만 교체
관성 텐서는 질량 비율만큼 선형으로 스케일합니다(형상·질량중심은 유지). 원본은 .bak로 저장됩니다.
"""
import argparse, csv, shutil, xml.etree.ElementTree as ET, os
here = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--urdf', default=os.path.join(here, '..', 'urdf', 'wheel_humanoid.urdf'))
g = ap.add_mutually_exclusive_group(required=True)
g.add_argument('--total', type=float)
g.add_argument('--csv')
a = ap.parse_args()
tree = ET.parse(a.urdf); root = tree.getroot()
links = {l.get('name'): l for l in root.findall('link')}
cur = {n: float(l.find('inertial/mass').get('value')) for n, l in links.items()}
if a.total:
    s = a.total / sum(cur.values()); target = {n: m * s for n, m in cur.items()}
else:
    target = {}
    for r in csv.DictReader(open(a.csv)):
        if r.get('cad_mass_kg', '').strip():
            target[r['link']] = float(r['cad_mass_kg'])
for n, m in target.items():
    k = m / cur[n]; ine = links[n].find('inertial')
    ine.find('mass').set('value', f'{m:.4f}')
    I = ine.find('inertia')
    for key in ('ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz'):
        I.set(key, f'{float(I.get(key)) * k:.6e}')
shutil.copy(a.urdf, a.urdf + '.bak')
tree.write(a.urdf, xml_declaration=True, encoding='utf-8')
print(f'updated {len(target)} links, total mass = {sum({**cur, **target}.values()):.3f} kg')
