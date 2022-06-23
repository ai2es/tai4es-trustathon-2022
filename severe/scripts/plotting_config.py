import numpy as np

deep_display_feature_names = {'ir069' : 'WV', 
                         'ir107' : 'IR', 
                         'vil'   : 'VIL',
                         'vis' : 'VIS',
                        }

deep_color_dict = {'ir107' : [1.0, 0.4980392156862745, 0.4980392156862745], 
              'ir069' : [0.49411764705882355, 0.5137254901960784, 0.9725490196078431],
              'vil' : 'y',
              'vis' : 'k',
             }




pretty_names = [ '$IR_{min}$', '$IR_{1st}$', '$IR_{10th}$', '$IR_{25th}$', '$IR_{med}$',
  '$IR_{75th}$',  '$IR_{90th}$',  '$IR_{99th}$', '$IR_{max}$', '$WV_{min}$', '$WV_{1st}$', '$WV_{10th}$', '$WV_{25th}$', '$WV_{med}$',
  '$WV_{75th}$',  '$WV_{90th}$',  '$WV_{99th}$', '$WV_{max}$', '$VIS_{min}$', '$VIS_{1st}$', '$VIS_{10th}$', '$VIS_{25th}$', '$VIS_{med}$',
  '$VIS_{75th}$',  '$VIS_{90th}$',  '$VIS_{99th}$', '$VIS_{max}$', '$VIL_{min}$', '$VIL_{1st}$', '$VIL_{10th}$', '$VIL_{25th}$', '$VIL_{med}$',
  '$VIL_{75th}$',  '$VIL_{90th}$',  '$VIL_{99th}$', '$VIL_{max}$',]

display_feature_names = {'q000_ir': '$IR_{min}$',
 'q001_ir': '$IR_{1st}$',
 'q010_ir': '$IR_{10th}$',
 'q025_ir': '$IR_{25th}$',
 'q050_ir': '$IR_{med}$',
 'q075_ir': '$IR_{75th}$',
 'q090_ir': '$IR_{90th}$',
 'q099_ir': '$IR_{99th}$',
 'q100_ir': '$IR_{max}$',
 'q000_wv': '$WV_{min}$',
 'q001_wv': '$WV_{1st}$',
 'q010_wv': '$WV_{10th}$',
 'q025_wv': '$WV_{25th}$',
 'q050_wv': '$WV_{med}$',
 'q075_wv': '$WV_{75th}$',
 'q090_wv': '$WV_{90th}$',
 'q099_wv': '$WV_{99th}$',
 'q100_wv': '$WV_{max}$',
 'q000_vi': '$VIS_{min}$',
 'q001_vi': '$VIS_{1st}$',
 'q010_vi': '$VIS_{10th}$',
 'q025_vi': '$VIS_{25th}$',
 'q050_vi': '$VIS_{med}$',
 'q075_vi': '$VIS_{75th}$',
 'q090_vi': '$VIS_{90th}$',
 'q099_vi': '$VIS_{99th}$',
 'q100_vi': '$VIS_{max}$',
 'q000_vl': '$VIL_{min}$',
 'q001_vl': '$VIL_{1st}$',
 'q010_vl': '$VIL_{10th}$',
 'q025_vl': '$VIL_{25th}$',
 'q050_vl': '$VIL_{med}$',
 'q075_vl': '$VIL_{75th}$',
 'q090_vl': '$VIL_{90th}$',
 'q099_vl': '$VIL_{99th}$',
 'q100_vl': '$VIL_{max}$'}

cs = ['q000_ir',
 'q001_ir',
 'q010_ir',
 'q025_ir',
 'q050_ir',
 'q075_ir',
 'q090_ir',
 'q099_ir',
 'q100_ir',
 'q000_wv',
 'q001_wv',
 'q010_wv',
 'q025_wv',
 'q050_wv',
 'q075_wv',
 'q090_wv',
 'q099_wv',
 'q100_wv',
 'q000_vi',
 'q001_vi',
 'q010_vi',
 'q025_vi',
 'q050_vi',
 'q075_vi',
 'q090_vi',
 'q099_vi',
 'q100_vi',
 'q000_vl',
 'q001_vl',
 'q010_vl',
 'q025_vl',
 'q050_vl',
 'q075_vl',
 'q090_vl',
 'q099_vl',
 'q100_vl']

r = [255/255,127/255,127/255]
b = [126/255,131/255,248/255]

colors = [r,b,'k','y']
color_dict = {}
cit = -1
for i in np.arange(0,36):
    if np.mod(i,9) == 0:
        cit += 1
        c = colors[cit]
    color_dict[cs[i]] = c