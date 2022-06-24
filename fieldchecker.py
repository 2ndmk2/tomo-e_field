#!/usr/bin/env python
# -*- coding: utf-8 *-

## Author; Ryou Ohsawa
from astropy.wcs import WCS
from astropy.coordinates import Longitude, Latitude, Angle, SkyCoord
import pandas as pd
import numpy as np
import io

from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
__pix2foc_str__ = '''det_id,pix_x,pix_y,foc_x,foc_y
111,1,1,2738.223088235294,1758.0011764705882
111,1,1128,2739.771323529412,631.0288235294121
111,2000,1,739.10175,1755.1083823529416
111,2000,1128,740.6019705882353,628.1850588235296
112,1,1,2746.056911764706,4181.766617647058
112,1,1128,2743.665294117647,3054.7502941176476
112,2000,1,746.7271764705882,4185.929705882353
112,2000,1128,744.3988676470588,3058.9016176470586
113,1,1,2755.1463235294123,6612.451323529412
113,1,1128,2754.6251470588236,5485.4895588235295
113,2000,1,755.4585147058822,6613.174411764708
113,2000,1128,755.2025147058825,5486.202205882355
114,1,1,2756.2714705882354,9034.82
114,1,1128,2754.194705882354,7907.762500000002
114,2000,1,756.1944999999998,9038.562500000002
114,2000,1128,754.374882352941,7911.562647058824
115,1,1,2764.4123529411763,11457.41176470588
115,1,1128,2763.748235294118,10330.352941176472
115,2000,1,763.6809999999997,11458.627941176468
115,2000,1128,763.3315147058827,10331.552941176471
116,1,1,2768.223235294118,13881.669117647056
116,1,1128,2766.684852941176,12754.68823529412
116,2000,1,766.7839264705882,13884.501470588231
116,2000,1128,765.6386617647058,12757.489705882352
121,1,1,6011.723676470589,1746.2566176470589
121,1,1128,6006.919558823529,619.1684852941177
121,2000,1,4012.5777941176475,1754.4814705882352
121,2000,1128,4007.9170588235293,627.4848382352943
122,1,1,6024.342205882354,4170.87338235294
122,1,1128,6017.302352941178,3043.7504411764703
122,2000,1,4025.054117647059,4183.465000000003
122,2000,1128,4018.1702941176472,3056.4063235294125
123,1,1,6028.424117647059,6601.25176470588
123,1,1128,6026.354117647058,5474.022205882352
123,2000,1,4028.7927941176476,6605.165147058827
123,2000,1128,4026.987205882354,5478.082058823529
124,1,1,6034.255735294118,9032.474999999993
124,1,1128,6032.618970588235,7905.253382352939
124,2000,1,4034.108676470591,9035.88117647059
124,2000,1128,4032.7379411764696,7908.793235294117
125,1,1,6038.446470588236,11445.744117647057
125,1,1128,6034.504852941177,10318.595588235297
125,2000,1,4037.850588235293,11453.567647058822
125,2000,1128,4034.2680882352947,10326.470588235296
126,1,1,6047.528088235294,13881.383823529413
126,1,1128,6045.42147058824,12754.314705882352
126,2000,1,4046.204117647057,13886.373529411767
126,2000,1128,4044.4216176470586,12759.31176470588
131,1,1,9283.422352941176,1741.8557352941173
131,1,1128,9282.36941176471,614.4122352941176
131,2000,1,7284.478823529413,1743.6597058823534
131,2000,1128,7283.396029411763,616.4289264705883
132,1,1,9287.561764705883,4174.504411764706
132,1,1128,9285.88926470588,3047.116176470589
132,2000,1,7288.357352941173,4177.855147058824
132,2000,1128,7286.837794117649,3050.499411764705
133,1,1,9296.70573529412,6606.157647058824
133,1,1128,9296.725147058823,5478.620294117649
133,2000,1,7297.111176470587,6606.825294117648
133,2000,1128,7297.413970588234,5479.498088235292
134,1,1,9305.374411764706,9030.05352941176
134,1,1128,9305.780441176472,7902.545735294119
134,2000,1,7305.382499999999,9030.354411764705
134,2000,1128,7306.0551470588225,7903.10955882353
135,1,1,9317.36867647059,11450.051470588236
135,1,1128,9313.309999999998,10322.679411764708
135,2000,1,7316.851029411765,11458.995588235297
135,2000,1128,7313.062794117649,10331.732352941179
141,1,1,12555.738235294115,1738.7389705882351
141,1,1128,12553.973529411765,610.8375588235291
141,2000,1,10556.757352941177,1741.8023529411764
141,2000,1128,10555.061764705883,614.1976617647059
142,1,1,12570.46176470588,4167.4864705882355
142,1,1128,12571.995588235295,3039.6005882352943
142,2000,1,10571.292647058828,4165.259117647059
142,2000,1128,10573.013235294118,3037.6379411764706
143,1,1,12574.20294117647,6584.865441176467
143,1,1128,12570.886764705887,5456.995294117648
143,2000,1,10574.855882352938,6591.974852941177
143,2000,1128,10571.627941176468,5464.316617647061
144,1,1,12573.039705882356,9014.960294117642
144,1,1128,12569.058823529413,7887.068529411764
144,2000,1,10573.364705882355,9023.920441176471
144,2000,1128,10569.577941176472,7896.335294117648
211,1,1,-770.0402499999999,1772.804117647059
211,1,1128,-774.8755147058824,645.9078088235293
211,2000,1,-2769.1952941176473,1781.1435294117641
211,2000,1128,-2773.8599999999997,654.2341323529411
212,1,1,-771.9070294117644,4198.575000000001
212,1,1128,-768.0467058823529,3071.6649999999995
212,2000,1,-2771.285294117647,4191.3839705882365
212,2000,1128,-2767.238382352941,3064.3872058823526
213,1,1,-769.7306764705879,6612.576911764705
213,1,1128,-772.9501911764708,5485.606617647058
213,2000,1,-2769.4866176470587,6618.002499999999
213,2000,1128,-2772.5605882352947,5491.03205882353
214,1,1,-766.2003235294117,9049.901176470587
214,1,1128,-763.7808235294115,7922.953235294117
214,2000,1,-2766.279264705883,9045.296029411766
214,2000,1128,-2763.636029411765,7918.225882352943
215,1,1,-766.5857647058824,11473.541176470584
215,1,1128,-767.7748529411762,10346.547058823528
215,2000,1,-2767.2769117647053,11474.973529411764
215,2000,1128,-2768.1677941176476,10347.863235294117
216,1,1,-763.4798382352944,13900.514705882353
216,1,1128,-765.5965294117647,12773.536764705881
216,2000,1,-2764.903823529413,13903.26617647059
216,2000,1128,-2766.617647058824,12776.223529411769
221,1,1,-4035.3232352941172,1771.0016176470594
221,1,1128,-4043.3677941176456,644.0407647058822
221,2000,1,-6034.522499999999,1785.0607352941183
221,2000,1128,-6042.520588235294,658.0250882352943
222,1,1,-4035.190441176471,4192.138823529413
222,1,1128,-4044.1436764705877,3065.1402941176475
222,2000,1,-6034.578676470591,4207.549117647059
222,2000,1128,-6043.383676470589,3080.437205882353
223,1,1,-4035.748529411765,6614.7520588235275
223,1,1128,-4040.314852941177,5487.68
223,2000,1,-6035.515588235292,6622.017205882353
223,2000,1128,-6039.821470588236,5494.836617647061
224,1,1,-4037.3580882352944,9044.697941176471
224,1,1128,-4041.913676470588,7917.546617647059
224,2000,1,-6037.558823529414,9051.778823529414
224,2000,1128,-6041.914558823531,7924.479558823531
225,1,1,-4037.4732352941182,11476.401470588233
225,1,1128,-4038.7295588235284,10349.317647058822
225,2000,1,-6038.2360294117625,11477.032352941176
225,2000,1128,-6039.113382352941,10349.76617647059
226,1,1,-4039.4902941176474,13906.494117647058
226,1,1128,-4041.220294117647,12779.430882352946
226,2000,1,-6040.845,13907.819117647061
226,2000,1128,-6042.334852941175,12780.601470588232
231,1,1,-7308.968235294119,1777.773088235294
231,1,1128,-7310.694852941178,650.5211323529411
231,2000,1,-9308.282500000001,1780.643529411764
231,2000,1128,-9309.887941176468,653.15
232,1,1,-7309.050294117644,4202.844558823531
232,1,1128,-7309.671470588234,3075.581029411764
232,2000,1,-9308.53367647059,4203.419411764707
232,2000,1128,-9309.036176470587,3075.892205882352
233,1,1,-7306.173088235295,6620.402647058823
233,1,1128,-7310.033382352942,5493.013970588237
233,2000,1,-9305.929264705881,6626.194705882353
233,2000,1128,-9309.664264705882,5498.629264705882
234,1,1,-7308.023676470585,9046.482941176473
234,1,1128,-7309.271323529413,7919.07
234,2000,1,-9308.234852941177,9046.929264705883
234,2000,1128,-9309.213529411769,7919.348235294115
235,1,1,-7310.3475,11482.397058823524
235,1,1128,-7312.655441176473,10354.964705882356
235,2000,1,-9311.125735294117,11484.09705882353
235,2000,1128,-9313.14838235294,10356.520588235291
241,1,1,-10577.529411764703,1781.765294117647
241,1,1128,-10580.15,654.0911617647059
241,2000,1,-12576.86470588235,1786.366764705882
241,2000,1128,-12579.420588235298,658.4968529411765
242,1,1,-10576.592647058824,4204.768676470588
242,1,1128,-10577.961764705882,3077.0879411764695
242,2000,1,-12576.020588235298,4206.501911764706
242,2000,1128,-12577.308823529413,3078.5660294117647
243,1,1,-10584.111764705884,6633.7232352941155
243,1,1128,-10581.680882352943,5505.903088235295
243,2000,1,-12583.967647058826,6627.971911764708
243,2000,1128,-12581.348529411762,5499.921470588237
244,1,1,-10584.172058823531,9056.966323529414
244,1,1128,-10582.102941176472,7929.227058823529
244,2000,1,-12584.388235294118,9051.049117647057
244,2000,1128,-12582.127941176475,7923.032647058822
311,1,1,-788.6990147058824,-1117.686323529412
311,1,1128,-789.4593676470588,-2244.5655882352944
311,2000,1,-2787.7927941176467,-1116.2747058823531
311,2000,1128,-2788.482647058823,-2243.213382352941
312,1,1,-786.9883088235293,-3542.1602941176484
312,1,1128,-791.6307794117648,-4668.930882352941
312,2000,1,-2786.1651470588254,-3533.7338235294114
312,2000,1128,-2790.8763235294114,-4660.647205882351
313,1,1,-789.1667647058825,-5963.165147058823
313,1,1128,-794.2187205882351,-7089.943970588232
313,2000,1,-2788.538235294118,-5953.595588235292
313,2000,1128,-2793.755735294117,-7080.433529411762
314,1,1,-791.4673529411766,-8385.495588235293
314,1,1128,-792.9047205882353,-9512.481764705884
314,2000,1,-2791.2400000000002,-8382.518823529412
314,2000,1128,-2792.8124999999995,-9509.533382352938
315,1,1,-786.8865735294119,-10812.172058823533
315,1,1128,-786.4148676470589,-11939.132352941173
315,2000,1,-2787.326911764706,-10812.335294117647
315,2000,1128,-2787.1305882352935,-11939.370588235295
316,1,1,-786.9210588235292,-13242.10882352941
316,1,1128,-786.5016176470589,-14369.122058823532
316,2000,1,-2787.9125000000013,-13242.167647058823
316,2000,1128,-2787.799705882354,-14369.155882352945
321,1,1,-4056.394264705883,-1119.6338235294115
321,1,1128,-4056.1024999999977,-2246.5748529411767
321,2000,1,-6055.5363235294135,-1119.894705882353
321,2000,1128,-6055.2764705882355,-2246.9339705882358
322,1,1,-4055.651470588237,-3536.2613235294107
322,1,1128,-4052.993823529411,-4663.209411764705
322,2000,1,-6054.892647058823,-3540.3411764705875
322,2000,1128,-6052.231176470589,-4667.417352941175
323,1,1,-4052.0163235294117,-5972.332794117646
323,1,1128,-4054.71161764706,-7099.291323529413
323,2000,1,-6051.490588235296,-5966.57823529412
323,2000,1128,-6054.313676470591,-7093.623235294115
324,1,1,-4055.590441176469,-8405.260294117648
324,1,1128,-4058.8161764705883,-9532.310441176473
324,2000,1,-6055.370000000002,-8398.451911764705
324,2000,1128,-6058.7922058823515,-9525.598235294121
325,1,1,-4057.292500000001,-10810.270588235298
325,1,1128,-4052.9951470588235,-11937.280882352945
325,2000,1,-6057.403235294118,-10816.835294117647
325,2000,1128,-6053.346176470591,-11943.982352941177
326,1,1,-4062.1495588235293,-13249.925000000003
326,1,1128,-4061.8001470588224,-14376.954411764704
326,2000,1,-6063.027794117648,-13248.904411764704
326,2000,1128,-6063.019411764706,-14376.050000000005
331,1,1,-7325.926323529411,-1109.1319117647058
331,1,1128,-7324.665294117648,-2236.3397058823534
331,2000,1,-9325.118970588239,-1110.8194117647063
331,2000,1128,-9323.832794117649,-2238.148088235294
332,1,1,-7326.901323529411,-3540.3147058823533
332,1,1128,-7327.77205882353,-4667.492352941178
332,2000,1,-9326.181764705887,-3537.8061764705885
332,2000,1128,-9327.052058823525,-4665.234411764705
333,1,1,-7324.023970588236,-5968.28367647059
333,1,1128,-7327.795735294118,-7095.572058823531
333,2000,1,-9323.50205882353,-5960.138382352941
333,2000,1128,-9327.464852941175,-7087.515294117647
334,1,1,-7330.645735294118,-8390.452058823528
334,1,1128,-7329.640882352942,-9517.813235294117
334,2000,1,-9330.500735294117,-8390.349852941174
334,2000,1128,-9329.670441176473,-9517.862352941174
335,1,1,-7337.669264705883,-10827.738235294115
335,1,1128,-7337.8391176470595,-11955.061764705883
335,2000,1,-9338.013970588236,-10825.239705882357
335,2000,1128,-9338.39073529412,-11952.73382352941
341,1,1,-10596.766176470584,-1125.2875000000004
341,1,1128,-10601.01911764706,-2252.84955882353
341,2000,1,-12595.83382352941,-1116.8516176470587
341,2000,1128,-12600.12352941177,-2244.653970588235
342,1,1,-10598.172058823526,-3540.9647058823534
342,1,1128,-10600.730882352937,-4668.53455882353
342,2000,1,-12597.508823529417,-3534.937352941176
342,2000,1128,-12600.108823529414,-4662.807941176471
343,1,1,-10599.754411764701,-5967.464558823531
343,1,1128,-10601.220588235294,-7095.093676470587
343,2000,1,-12599.266176470592,-5963.045735294114
343,2000,1128,-12600.886764705881,-7090.968676470588
344,1,1,-10600.957352941175,-8400.645147058822
344,1,1128,-10602.4294117647,-9528.34544117647
344,2000,1,-12600.807352941172,-8395.491617647056
344,2000,1128,-12602.523529411767,-9523.46426470588
411,1,1,2749.6616176470593,-1135.0017647058821
411,1,1128,2744.0108823529413,-2261.861176470589
411,2000,1,750.6870588235296,-1125.2179411764698
411,2000,1128,745.0604558823528,-2252.0361764705885
412,1,1,2752.3197058823534,-3560.096470588236
412,1,1128,2748.8026470588225,-4687.030882352942
412,2000,1,753.1611029411763,-3554.0463235294114
412,2000,1128,749.6190588235294,-4680.866764705882
413,1,1,2747.126029411764,-5996.543382352941
413,1,1128,2738.22,-7123.342647058825
413,2000,1,747.8097647058823,-5981.00676470588
413,2000,1128,738.8685588235296,-7107.737647058826
414,1,1,2742.086323529412,-8402.223823529412
414,1,1128,2739.8310294117646,-9529.142647058823
414,2000,1,742.4120147058825,-8398.431470588235
414,2000,1128,739.890161764706,-9525.34088235294
415,1,1,2740.7117647058835,-10840.002941176474
415,1,1128,2736.108823529411,-11966.998529411763
415,2000,1,740.4526029411766,-10832.061764705883
415,2000,1128,735.4590588235294,-11959.013235294116
416,1,1,2729.214411764706,-13270.380882352938
416,1,1128,2731.1547058823526,-14397.413235294121
416,2000,1,728.2570147058823,-13274.233823529408
416,2000,1128,729.9159117647059,-14401.200000000004
421,1,1,6018.369264705885,-1143.9677941176465
421,1,1128,6016.270882352941,-2271.085441176471
421,2000,1,4019.3138235294105,-1140.6201470588232
421,2000,1128,4017.2073529411755,-2267.5773529411767
422,1,1,6015.059705882353,-3567.079411764706
422,1,1128,6012.688235294116,-4694.172352941176
422,2000,1,4015.9082352941186,-3563.457352941176
422,2000,1128,4013.493970588234,-4690.4286764705885
423,1,1,6012.138823529409,-5996.42705882353
423,1,1128,6007.096029411766,-7123.514264705884
423,2000,1,4012.7672058823537,-5988.0505882352945
423,2000,1128,4007.551764705882,-7115.006323529412
424,1,1,6011.056865671641,-8415.110895522388
424,1,1128,6004.037313432835,-9542.23626865672
424,2000,1,4011.315820895522,-8403.602089552236
424,2000,1128,4004.1140298507466,-9530.539402985074
425,1,1,6002.982205882354,-10843.413235294112
425,1,1128,6002.123676470589,-11970.592647058822
425,2000,1,4002.703676470588,-10842.88088235294
425,2000,1128,4001.4835294117656,-11969.942647058826
426,1,1,6006.585294117647,-13286.03676470588
426,1,1128,5999.87911764706,-14413.173529411764
426,2000,1,4005.586029411765,-13275.716176470589
426,2000,1128,3998.5173529411763,-14402.726470588233
431,1,1,9283.673235294118,-1134.185147058824
431,1,1128,9287.955588235298,-2261.6245588235297
431,2000,1,7284.770735294119,-1142.3170588235294
431,2000,1128,7288.981911764707,-2269.474558823529
432,1,1,9280.581323529412,-3579.703823529413
432,1,1128,9279.702794117644,-4707.140735294118
432,2000,1,7281.450735294118,-3579.1120588235294
432,2000,1128,7280.465147058825,-4706.249558823528
433,1,1,9288.870000000004,-5994.734264705883
433,1,1128,9288.035294117646,-7122.185735294121
433,2000,1,7289.480882352938,-5994.379705882353
433,2000,1128,7288.538088235294,-7121.673676470587
434,1,1,9283.597647058823,-8427.48455882353
434,1,1128,9274.471323529413,-9554.894705882354
434,2000,1,7283.850441176472,-8412.576029411764
434,2000,1128,7274.472941176469,-9539.815147058822
435,1,1,9278.25808823529,-10844.238235294119
435,1,1128,9279.060147058819,-11971.66470588235
435,2000,1,7277.963823529412,-10847.561764705883
435,2000,1128,7278.544411764707,-11974.838235294117
441,1,1,12556.90735294118,-1155.7725
441,1,1128,12552.076470588243,-2283.6326470588237
441,2000,1,10558.047058823531,-1147.71955882353
441,2000,1128,10553.08823529412,-2275.2820588235295
442,1,1,12552.605882352944,-3578.4580882352943
442,1,1128,12547.93088235294,-4706.355588235293
442,2000,1,10553.505882352938,-3571.1355882352937
442,2000,1128,10548.66764705882,-4698.664558823529
443,1,1,12554.047058823533,-5998.614117647058
443,1,1128,12552.588235294117,-7126.556764705882
443,2000,1,10554.632352941173,-5997.5517647058805
443,2000,1128,10553.051470588238,-7125.226323529414
444,1,1,12556.316176470587,-8428.372500000001
444,1,1128,12556.01470588236,-9556.332647058818
444,2000,1,10556.574999999999,-8430.007205882353
444,2000,1128,10556.064705882352,-9557.59411764706
'''


def LongitudeLike(arg):
  try:
    # assuming the argument is float in degree
    arg = float(arg)
    return Longitude(arg, unit='degree')
  except:
    # assuming the argument is string, if failed
    assert isinstance(arg, str)
    if ':' in arg:
      return Longitude(arg, unit='hourangle')
    else:
      return Longitude(arg)


def LatitudeLike(arg):
  try:
    # assuming the argument is float in degree
    arg = float(arg)
    return Latitude(arg, unit='degree')
  except:
    # assuming the argument is string, if failed
    assert isinstance(arg, str)
    if ':' in arg:
      return Latitude(arg, unit='degree')
    else:
      return Latitude(arg)


def Arcsecond(arg):
  return Angle(arg, unit='arcsec')


def Radian(arg):
  return Angle(arg, unit='radian')


def get_projection(ra,dec,pixscale):
  ''' Obtain WCS object

  Parameters:
    ra   (Longitude): right ascension at the center of the focal plane
    dec   (Latitude): declination at the center of the focal plane
    pixscale (Angle): pixel scale at the center of the focal plane
  '''
  w = WCS()
  w.wcs.ctype = 'RA---ARC', 'DEC--ARC'
  w.wcs.crval = [ra.degree, dec.degree]
  w.wcs.crpix = [1.0, 1.0]
  w.wcs.cdelt = [-pixscale.degree, pixscale.degree]
  return w


def generate_footprint(xy):
  ''' Generate Detector Footprint

  Parameters:
    xy (ndarray):
      vertices coordinates of a detector.
  '''
  xy = np.array(xy)[np.array([0,1,3,2])]
  #xy = np.array(xy)[np.array([0,1,3,2])]
  return Polygon(xy, color='orange', alpha=0.5)


def detector_footprint(df):
  ''' Generate the footprints of the detectors

  Parameters:
    df (DataFrame):
      dataframe of the detector coordinate on the forcal plane.
  '''
  det_id = df['det_id'].unique()
  detectors = [
    generate_footprint(df.loc[df.det_id==d,['foc_x','foc_y']]) for d in det_id]
  return PatchCollection(detectors,
      linewidth=0.5, edgecolor='k', facecolor=(0.3,0.3,0.3,0.1))


def calc_deltaphi(foc_x, foc_y, pixscale):
  ''' Calculate the angular distance and position angle

  Parameters:
    foc_x (float):
      x-coordinate of the target point on the focal plane.
    foc_y (float):
      y-coordinate of the target point on the focal plane.
    pixscale (Angle):
      pixel scale at the center of the focal plane.

  Return:
    delta (Angle):
      the angular distance toward the target point.
    pa (Angle):
      the position angle toward the target point.
      the north is the origin. the `pa` increase to the east.
  '''
  foc_x,foc_y = np.array(foc_x), np.array(foc_y)
  delta = Arcsecond(pixscale*np.sqrt(foc_x**2+foc_y**2))
  pa = Radian(np.arctan2(-foc_x,foc_y))
  return delta, pa


def calc_pointing(ra, dec, delta, phi):
  ''' Calculate the optimal pointing position.

  Parameters:
    ra (Longitude):
      right ascension of the target point.
    dec (Latitude):
      declination of the target point.
    delta (Angle):
      angular distance between the origin and the target point.
    phi (Angle):
      position angle toward the target point from the origin.

  Return:
    ra (Longitude):
      right ascension of the optmail pointing.
    dec (Latitude):
      declination of the optimal pointing.
  '''
  da = -Radian(np.arcsin(np.sin(phi.rad)*np.sin(delta.rad)/np.cos(dec.rad)))
  L = np.sqrt(np.cos(delta.rad)**2+np.cos(phi.rad)**2*np.sin(delta.rad)**2)
  t = Radian(np.arctan2(np.cos(phi.rad)*np.sin(delta.rad),np.cos(delta.rad)))
  dec = -t + Radian(np.arcsin(np.sin(dec.rad)/L))
  return Longitude(ra+da).to('hourangle'),Latitude(dec).to('degree')


if __name__ == '__main__':
  from argparse import ArgumentParser as ap
  parser = ap()

  parser.add_argument(
    'ra_tel', type=LongitudeLike,
    help='Right Ascension of the telescope pointing')
  parser.add_argument(
    'dec_tel', type=LatitudeLike,
    help='Declination of the telescope pointing')
  parser.add_argument(
    'ra_obj', type=LongitudeLike, nargs='?',
    help='Right Ascension of the object')
  parser.add_argument(
    'dec_obj', type=LatitudeLike, nargs='?',
    help='Declination of the object')
  parser.add_argument(
    '--det_id', type=int,
    help='highlight detector by detector ID')
  parser.add_argument(
    '--pixscale', type=Arcsecond, default=Arcsecond(1.189),
    help='pixel scale at the center of the focal plane')
  parser.add_argument(
    '--crpix', type=str, action='store', default=None,
    help='data table for pixel locations on the focal plane')
  parser.add_argument(
    '--delta_ra', type=float, action='store', default=0.0,
    help='offset delta between the optical center and geometric center')
  parser.add_argument(
    '--delta_dec', type=float, action='store', default=0.0,
    help='offset delta between the optical center and geometric center')

  args = parser.parse_args()
  if args.crpix is None:
    csv = io.StringIO(__pix2foc_str__)
  else:
    csv = args.crpix
  df = pd.read_csv(csv)
  gc = df[['foc_x','foc_y']].mean()
  gc.foc_x += args.delta_ra
  gc.foc_y += args.delta_dec

  delta,phi = calc_deltaphi(gc.foc_x, gc.foc_y, args.pixscale)
  ra_opt,dec_opt = calc_pointing(
    args.ra_tel, args.dec_tel, delta, phi)
  proj = get_projection(ra_opt, dec_opt, args.pixscale)

  from matplotlib.patches import Polygon
  from matplotlib.collections import PatchCollection
  import matplotlib.pyplot as plt

  detectors = detector_footprint(df)

  focpix = np.array(df[['foc_x', 'foc_y']]).reshape([-1,2])
  coord = SkyCoord(50, 50, frame='icrs', unit='deg')
  print(proj.all_world2pix([[50,50]], 0))

  fig,ax = plt.subplots(figsize=(10,10), subplot_kw={'projection': proj})
  ax.add_collection(detectors)
  ax.scatter(0.0, 0.0, marker='x', label='optical center')
  ax.scatter(gc['foc_x'], gc['foc_y'], marker='x', label='gravity center')
  ax.scatter(focpix[:,0],focpix[:,1],10,color='k',marker='.')
  if args.det_id is not None:
    highlight = generate_footprint(
      df.loc[df.det_id==args.det_id,['foc_x','foc_y']])
    ax.add_patch(highlight)
  if args.ra_obj is not None and args.dec_obj is not None:
    target = np.array([args.ra_obj.degree,args.dec_obj.degree]).reshape([-1,2])
    target = np.array(proj.all_world2pix(target, 0)).reshape([-1,2])
    ax.scatter(target[0,0], target[0,1], 100,
               linewidth=1, marker='x', color='r', label='target')
    if args.det_id is not None:
      det = df.loc[df.det_id==args.det_id,:]
      oz  = det.loc[(det.pix_x==1) & (det.pix_y==1),:]
      ax1 = det.loc[(det.pix_x==2000) & (det.pix_y==1),:]
      ax2 = det.loc[(det.pix_x==1) & (det.pix_y==1128),:]
      p = target - np.array(oz[['foc_x','foc_y']]).reshape([-1,2])
      d1 = np.array(ax1[['foc_x','foc_y']])-np.array(oz[['foc_x','foc_y']])
      d2 = np.array(ax2[['foc_x','foc_y']])-np.array(oz[['foc_x','foc_y']])
      A = np.vstack([d1/1999,d2/1127]).T
      pix = np.linalg.solve(A,p.flat)
      print(f'detector {args.det_id}')
      print(f'  NAXIS1: {pix[0]+1}')
      print(f'  NAXIS2: {pix[1]+1}')

  ax.set_xlabel('Right Ascension', fontsize=16)
  ax.set_ylabel('Declination', fontsize=16)
  ax.grid()
  ax.legend(loc='upper right')
  plt.show()