import numpy as np
import pylatticeio as io

latt_size = [4, 4, 4, 8]

gauge_lime = io.readChromaQIOGauge("./data/weak_field.lime")
gauge_kyu = io.readKYUGauge("./data/weak_field.kyu", latt_size)
print(np.linalg.norm(gauge_lime - gauge_kyu))

io.writeKYUGauge("weak_field.2.kyu", latt_size, gauge_lime)
gauge_kyu_tmp = io.readKYUGauge("./weak_field.2.kyu", latt_size)
print(np.linalg.norm(gauge_kyu_tmp - gauge_kyu))

prop_lime = io.readChromaQIOPropagator("./data/pt_prop_1")
prop_kyu = io.readKYUPropagator("./data/pt_prop_1.kyu", latt_size)
prop_kyu_f = io.readKYUPropagatorF("./data/pt_prop_1.kyu_f", latt_size)
print(np.linalg.norm(prop_lime - prop_kyu))
print(np.linalg.norm(prop_lime - prop_kyu_f))

io.writeKYUPropagator("pt_prop_1.2.kyu", latt_size, prop_lime)
io.writeKYUPropagatorF("pt_prop_1.2.kyu_f", latt_size, prop_lime)
prop_kyu_tmp = io.readKYUPropagator("./pt_prop_1.2.kyu", latt_size)
prop_kyu_f_tmp = io.readKYUPropagatorF("./pt_prop_1.2.kyu_f", latt_size)
print(np.linalg.norm(prop_kyu_tmp - prop_kyu))
print(np.linalg.norm(prop_kyu_f_tmp - prop_kyu_f))
