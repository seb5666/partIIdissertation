import matplotlib.pyplot as plt

import numpy as np

from matplotlib2tikz import save as tikz_save

#82/40
#history = {'loss': [1.1834619284481473, 0.8779414423539903, 0.78742955471038822, 0.72972224728266399, 0.69273273747973974, 0.6646335147857666, 0.64205221101548937, 0.62045197785483464, 0.60306351870430841, 0.58704581855350069, 0.57218706055323287, 0.55987769703759083, 0.54961887247509422, 0.53798246742672395, 0.53034748007880317, 0.52253594941457115, 0.51620738610373607, 0.51020303045484749, 0.50411822373920012, 0.49963104018635218, 0.49532787332746719, 0.48945741917292279, 0.48451624891493056, 0.4809836290825738, 0.48006195536719426, 0.47423566709306503, 0.4754476164563497, 0.46888625390582617, 0.46744501386748422, 0.46465126547283597, 0.46138486713409421, 0.46253726328743827, 0.45809416914410062, 0.45632927580727473, 0.4543125896114773, 0.45376535197152029, 0.4533875278133816, 0.450039047115114, 0.45206780813641018, 0.45172852335611979],'val_acc': [0.62559999799728394, 0.68040000104904175, 0.70240000152587889, 0.72520000219345093, 0.74200000143051148, 0.74879999685287479, 0.75919999837875363, 0.7727999968528747,0.77879999876022343, 0.78720000028610226, 0.79160000133514408, 0.79560000133514408, 0.79240000057220461, 0.80479999828338622, 0.80679999828338622, 0.80440000247955323, 0.81080000209808345, 0.81119999790191655, 0.81320000314712526, 0.82080000209808346, 0.82279999971389772, 0.81920000123977665, 0.82519999933242794, 0.82039999818801879, 0.82119999790191656, 0.82560000085830687, 0.82720000171661379, 0.82359999895095826, 0.82279999589920039, 0.83000000619888303, 0.8312000036239624, 0.83079999780654912, 0.82880000162124634, 0.8292000069618225, 0.82800000047683719, 0.83280000543594357, 0.83120000123977666, 0.83160000085830688, 0.83200000429153442, 0.83200000238418581], 'acc': [0.50007555555555561, 0.6541688888846503, 0.69340444444868299, 0.7162977777735392, 0.73062000000423855, 0.74150888889312749, 0.75001555555767485, 0.75709777777989706, 0.76475111111111116, 0.77132222222222224, 0.77645333333757194, 0.78110000000211932, 0.78582222222646081, 0.79120444444656368, 0.79415777777565855, 0.79769777777353923, 0.80001111110687251, 0.80291333333333337, 0.80529777777353928, 0.80730444444868299, 0.80870444444020584, 0.81163333333757193, 0.81405555555555553, 0.81481999999788068, 0.81534444444868304, 0.81690888889312741,0.81753333332909484, 0.82017777778201628, 0.82083111110687257, 0.82129333333545262, 0.82297555555555557, 0.82187777777353921, 0.82392888889312743, 0.82451999999576142, 0.82501111111111114, 0.82586666666454744, 0.82628666666454742, 0.82742888889312749, 0.8267822222179837, 0.8265999999957615], 'val_loss': [0.94137694644927983, 0.81970320749282832, 0.7563410530090332, 0.70901611566543576, 0.68497156620025634, 0.65031453990936283, 0.64089205646514891, 0.60883078527450563, 0.5983222970962524, 0.5781377882957458, 0.57214120674133295, 0.56135089445114139, 0.56469029092788692, 0.53448022532463069, 0.51948575210571291, 0.52563897967338558, 0.5063207678794861, 0.49801317977905274, 0.5062009332180023, 0.48837529826164244, 0.48510376977920533, 0.48245595788955686, 0.47615435171127318, 0.47863467049598696, 0.47473221802711485, 0.4700774145126343, 0.46420446825027467, 0.47236983013153078, 0.46823214769363403, 0.46066839623451233, 0.45748015260696412, 0.45709354615211489, 0.45764015436172484, 0.45372467136383055, 0.45833430600166319, 0.45064549279212951, 0.45020099067687991, 0.4501874997615814, 0.44969335412979128, 0.44845229506492612]}

#83
#history = {'val_acc': [0.65200000238418576, 0.68720000457763675, 0.71680000305175784, 0.73640000057220456, 0.74000000667572019, 0.75879999971389767, 0.76080000209808352, 0.76840000391006469, 0.77159999942779545, 0.78360000276565556, 0.7816000070571899, 0.79079999971389769, 0.79480000114440919, 0.79359999895095823, 0.79320000171661376, 0.79200000095367429, 0.79320000362396237, 0.79759999895095823, 0.798399998664856,0.79799999523162846], 'loss': [1.0981167450480991, 0.83734875386979846, 0.7594238625250922, 0.71790604038874306, 0.68532154190487327, 0.66132536554972332, 0.63640161706076726, 0.62115192241880623, 0.60367336101955837, 0.59082401003943552, 0.57876306571324665, 0.57042986648135718, 0.56213328687455921, 0.55524355186886254, 0.54764935018115568, 0.54493611392762931, 0.54085069942792252, 0.53873926564322583,0.53509111435784229, 0.53544566212760081], 'val_loss': [0.91170815753936763, 0.8030092616081238, 0.73561338329315185, 0.69812980699539184, 0.67544270992279054, 0.63994818496704098, 0.62227248144149783,0.61498494863510134, 0.60298051643371586, 0.57942050552368163, 0.58059012985229497, 0.56146364402770998, 0.56626781606674192, 0.54829319000244137, 0.54483068275451663, 0.54355226612091068, 0.54042464685440061, 0.53622215843200682, 0.53412376308441167, 0.53319681358337401], 'acc': [0.54669777778201634, 0.6703755555555555, 0.70358888888465032, 0.72026000000211932, 0.73212444444444447, 0.74088888889312743, 0.75096222222010289, 0.75739333333333336, 0.76393777778201633, 0.76904444444232511, 0.77449111111323043, 0.77814666666666665, 0.78102666666242815, 0.78433555555555556, 0.78711333333121403, 0.78846000000423855, 0.79035999999576145, 0.79169555555131699, 0.79270666666242806, 0.79241333333757191]}

#84/40
history = {'loss': [0.68675335487365718, 0.42598207700729368, 0.34850509860038759, 0.3158262291622162, 0.29104889084815977, 0.27581829063415525, 0.26143240528106687, 0.25038514247894289, 0.24279407320022584, 0.2353300157070159, 0.22750268744468688, 0.2220149723625183, 0.21662488634109497, 0.21125764043807985, 0.20637870930194854, 0.20144002759933471, 0.20033477416038514, 0.19464069247722626, 0.19158207597732543, 0.18888827049255372, 0.18627101472854615, 0.18351480690956115, 0.1809907507610321, 0.17888891392707826, 0.17662728823661805, 0.17440956159591675, 0.17290547665596009, 0.17085449585914611, 0.1697549579048168, 0.16752440288543702, 0.16620338274955748, 0.16467637814521791, 0.1629490516281128, 0.16171758678913117, 0.16001373819351197, 0.15915356633186339, 0.15894083828926087, 0.15620538366317749, 0.1554565277698975, 0.15379771593093872], 'val_loss': [0.61744581403732302, 0.39628951668739321, 0.39021260313987732, 0.34359463372230531, 0.30134446275234222, 0.30491435370445252, 0.29911568250656129, 0.26482140939235688, 0.2508034191966057, 0.26056243219375608, 0.26119405763149262, 0.23455478587150574, 0.24978137862682342, 0.20715038130283356, 0.21811410152912139, 0.2198916897535324, 0.20597598569393158, 0.20898763594627381, 0.20224536430835724, 0.22917421927452086, 0.2122613308429718, 0.19672969315052033, 0.18621521558761597, 0.18658231215476989, 0.19364792915582657, 0.19197028982639314, 0.18799299983978271, 0.19097551889419556, 0.17896788175106049, 0.18695312402248382, 0.1709665608882904, 0.1696942400932312, 0.16906248428821563, 0.1715719125032425, 0.17756747270822526, 0.16508498722314835, 0.1640326602101326, 0.16550066658258439, 0.16684656655788421, 0.15907730689048766], 'acc': [0.72643765625000001, 0.83169625000000003, 0.86153828124999998, 0.87366468750000004, 0.88288484374999998, 0.88873171875000001, 0.89434578124999997, 0.89848843749999996, 0.90149609374999995, 0.90426937500000004, 0.90750546875000004, 0.90961031250000002, 0.91184796874999996, 0.91365281249999997, 0.91583328124999996, 0.91770218749999999, 0.91815484375, 0.92048937500000005, 0.92147328125000005, 0.92271796875000001, 0.92359968749999999, 0.92485984374999997, 0.92574765625, 0.92673843749999996, 0.92757171875, 0.92830171875, 0.92890218749999998, 0.92976046874999996, 0.93023859374999995, 0.93114078124999999, 0.93165078125, 0.93224953124999999, 0.93301250000000002, 0.93347999999999998, 0.93409156250000003, 0.93442359374999995, 0.93463140624999996, 0.93571484375000002, 0.93606515624999997, 0.93670468750000002], 'val_acc': [0.75788124999999995, 0.84418749999999998, 0.84883750000000002, 0.86181249999999998, 0.88005, 0.87815624999999997, 0.87858124999999998, 0.89347500000000002, 0.89894375000000004, 0.89612499999999995, 0.89429999999999998, 0.90389375000000005, 0.89938125000000002, 0.91534375000000001, 0.91261875000000003, 0.91160624999999995, 0.91636249999999997, 0.91664374999999998, 0.91835, 0.90816249999999998, 0.91396875, 0.91908124999999996, 0.92402499999999999, 0.92361249999999995, 0.9201125, 0.92206250000000001, 0.92295000000000005, 0.92114375000000004, 0.92597499999999999, 0.92371250000000005, 0.92983749999999998, 0.93029375000000003, 0.92976250000000005, 0.93028750000000004, 0.92695000000000005, 0.93215625000000002, 0.93259375, 0.93204374999999995, 0.93131874999999997, 0.93459999999999999]}

#85
history = {
    'val_acc': [0.84920000362396242, 0.83640000057220454, 0.84359999895095827, 0.849999997138977, 0.86959999370574947, 0.86080000019073488, 0.85719999790191648, 0.88320000028610235, 0.87160000324249265, 0.86560000133514403, 0.84560000324249263, 0.87559999418258672, 0.8788000001907349, 0.8691999998092651, 0.86200000095367435, 0.87400000095367436, 0.87360000276565553, 0.87319999790191649, 0.88159999799728395,0.87759999608993533],
    'val_loss': [0.39322190761566161, 0.4212889702320099, 0.37399686622619627, 0.38455916881561281, 0.34231582617759704, 0.3562952401638031, 0.35639283204078676, 0.31465220975875857, 0.3471987955570221, 0.35738116240501405, 0.39376532495021821, 0.32998534083366393, 0.3089012738466263, 0.32846895241737367, 0.34440674424171447, 0.33381610250473021, 0.33625777506828308, 0.31935319030284881,0.29806429827213288, 0.3098343930244446],
    'loss': [0.52616605752203205, 0.39775287428538003, 0.37103902959399754, 0.3565749130174849, 0.34580090175628664, 0.34116122292624579, 0.33656065155347187, 0.33359902522404988, 0.32793826428837247, 0.32731552561865912, 0.32319774664666917, 0.32033662851333616, 0.31900965500725642, 0.31876589511129594, 0.31570135916180081, 0.31396865289423198, 0.31386542101542153, 0.3127098318523831, 0.31257287807676526, 0.31029756253242491],
    'acc': [0.79748000000423858, 0.8491644444444445, 0.8594222222222222, 0.86511555555767483, 0.86959555555767487, 0.8710377777777778, 0.87312444444656367, 0.87427555555767478, 0.87617777778201633, 0.87699777778201637, 0.87842222222434152, 0.87872888888888889, 0.88012666666666661, 0.88055777777777777, 0.88126888888465038, 0.88224000000000002, 0.88192666666454744, 0.88274666666878598, 0.88289111111323038, 0.88408666666666669]}

#86
history = {
    'acc': [0.52908444444232516, 0.66010666666242812, 0.70007999999576143, 0.71887111111111113, 0.73214222222222225, 0.74271999999788074, 0.7499688888846503, 0.75836444444868301, 0.76488888888888884, 0.77083111111111113, 0.77605333333545257, 0.77905555555979411, 0.78311333333545263, 0.78537999999999997, 0.78866222222434146, 0.78859333333757187, 0.79159777777989704, 0.79298222222010295, 0.7935511111111111, 0.79433777777777781],
    'val_acc': [0.61840000057220457, 0.6735999999046326, 0.68720000123977665, 0.71039999771118167, 0.71919999694824222, 0.73159999847412105, 0.74319999885559085, 0.73800000333786009, 0.74759999847412106, 0.75439999914169309, 0.75920000076293948, 0.75960000181198117, 0.76599999761581417, 0.76360000038146969, 0.76840000295639033, 0.76360000228881841, 0.77440000104904172, 0.77040000295639033, 0.77320000219345097, 0.77360000181198119],
    'val_loss': [0.96376254749298096, 0.84249492979049678, 0.77617279911041259, 0.73650119018554683, 0.70885239648818965, 0.68199206829071046, 0.66255887889862064,0.65680161476135257, 0.63991705942153931, 0.62556092691421505, 0.60864027309417723, 0.60718484067916867, 0.59390695858001707, 0.58575641822814939, 0.58058564901351928, 0.58134495782852169, 0.57620149946212773, 0.57251770639419552, 0.5708002095222473, 0.5694663982391357],
    'loss': [1.130413912908766, 0.86106169440375435, 0.76787904865476819, 0.7182502047665914, 0.68715387994342381, 0.65751240613513517, 0.63839559453540373, 0.61914505914900042, 0.6031787941847907, 0.58809072117275663, 0.57666820398754548, 0.56690747420840792, 0.55915147215949168, 0.55212442444907295, 0.54601055546442667, 0.54380677711698744, 0.5381342763731215, 0.53504091784159347, 0.53354985645294184, 0.531543177839915]}

x = np.arange(1, len(history['loss']) + 1)
plt.plot(x, history['val_loss'], '-',  label="validation loss")
plt.plot(x, history['loss'], '-',  label="loss")
plt.plot(x, history['val_acc'], '-',  label="valdiation accuracy")
plt.plot(x, history['acc'], '-',  label="accuracy")
plt.xlabel('Training epoch')
plt.legend(loc='upper right')
plt.xlim((1,len(history['loss'])))
tikz_save('../plots/pereira_validation_accuracy.tex', figureheight = '\\figureheight', figurewidth = '\\figurewidth')
