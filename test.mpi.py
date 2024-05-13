import numpy as np
import pylatticeio as io

io.setGrid([1, 1, 1, 2])

gauge_lime = io.readChromaQIOGauge("./data/weak_field.lime")
gauge_kyu = io.readKYUGauge("./data/weak_field.kyu", [4, 4, 4, 8])
print(np.linalg.norm(gauge_lime - gauge_kyu))

io.writeKYUGauge("weak_field.2.kyu", gauge_lime)
gauge_kyu_tmp = io.readKYUGauge("./weak_field.2.kyu", [4, 4, 4, 8])
print(np.linalg.norm(gauge_kyu_tmp - gauge_kyu))

prop_lime = io.readChromaQIOPropagator("./data/pt_prop_1")
prop_kyu = io.readKYUPropagator("./data/pt_prop_1.kyu", [4, 4, 4, 8])
prop_kyu_f = io.readKYUPropagatorF("./data/pt_prop_1.kyu_f", [4, 4, 4, 8])
print(np.linalg.norm(prop_lime - prop_kyu))
print(np.linalg.norm(prop_lime - prop_kyu_f))

io.writeKYUPropagator("pt_prop_1.2.kyu", prop_lime)
io.writeKYUPropagatorF("pt_prop_1.2.kyu_f", prop_lime)
prop_kyu_tmp = io.readKYUPropagator("./pt_prop_1.2.kyu", [4, 4, 4, 8])
prop_kyu_f_tmp = io.readKYUPropagatorF("./pt_prop_1.2.kyu_f", [4, 4, 4, 8])
print(np.linalg.norm(prop_kyu_tmp - prop_kyu))
print(np.linalg.norm(prop_kyu_f_tmp - prop_kyu_f))
