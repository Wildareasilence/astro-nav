"""下载 Gaia DR3 子样本（近星 + 全天亮星）。只需跑一次。

    python scripts/00_download_catalog.py

近星样本：parallax>5.5 mas（180 pc 内）、视差信噪比>5、G<12
亮星样本：G<8.5，全天
两个合起来才有足够的三维距离信息和足够的亮星供配对/定向。
"""
import os
import sys
import time
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as CF

TAP = 'https://gea.esac.esa.int/tap-server/tap/sync'
COLS = ('source_id,ra,dec,parallax,parallax_error,pmra,pmdec,'
        'radial_velocity,phot_g_mean_mag,bp_rp,ruwe')


def query(adql, out_path, tries=4, timeout=180):
    data = urllib.parse.urlencode({'REQUEST': 'doQuery', 'LANG': 'ADQL',
                                   'FORMAT': 'csv', 'QUERY': adql}).encode()
    for t in range(tries):
        try:
            with urllib.request.urlopen(TAP, data=data, timeout=timeout) as r:
                body = r.read().decode('utf-8', 'ignore')
            if body.count(',') > 100 and 'ERROR' not in body[:200]:
                open(out_path, 'w', encoding='utf-8').write(body)
                return body.count('\n') - 1
        except Exception as e:
            print(f'    第 {t + 1} 次失败：{type(e).__name__}')
        time.sleep(4)
    return 0


def fetch(where, prefix, nslice=12):
    for i in range(nslice):
        a, b = i * 360 // nslice, (i + 1) * 360 // nslice
        f = os.path.join(CF.CATALOG_DIR, f'{prefix}_{i}.csv')
        if os.path.exists(f) and open(f, encoding='utf-8', errors='ignore').readline().count(',') > 5:
            print(f'  {prefix}_{i}: 已有 {sum(1 for _ in open(f)) - 1} 行')
            continue
        q = (f'SELECT {COLS} FROM gaiadr3.gaia_source WHERE {where} '
             f'AND ra>={a} AND ra<{b}')
        n = query(q, f)
        print(f'  {prefix}_{i}: {n} 行')


def main():
    print('近星样本（d<180 pc，G<12）')
    fetch('parallax>5.5 AND parallax_over_error>5 AND phot_g_mean_mag<12', 'part')
    print('全天亮星（G<8.5）')
    fetch('phot_g_mean_mag<8.5', 'bb')
    tot = sum(sum(1 for _ in open(os.path.join(CF.CATALOG_DIR, f))) - 1
              for f in os.listdir(CF.CATALOG_DIR) if f.endswith('.csv'))
    print(f'\n合计 {tot} 行，位于 {CF.CATALOG_DIR}')


if __name__ == '__main__':
    main()
