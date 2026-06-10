
from src.models.heatpump_model import HEATPUMPS

hp = HEATPUMPS["idm"]()

print(hp.compute_COP(2, 35))