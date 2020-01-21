# Stability search for spindles

# rm -R SearchSpindles

# python run.py --folder SearchSpindles --params params_Th.py --search search.py --map yes nest
# python run.py --folder SearchTh2 --params params_Th.py --search search2.py --map yes nest


python run.py --folder SearchPY_EPSP --params PY_epsp_response.py --search search.py --map yes nest
python run.py --folder SearchINH_EPSP --params INH_epsp_response.py --search search.py --map yes nest
python run.py --folder SearchPY_IPSP --params PY_ipsp_response.py --search search.py --map yes nest
python run.py --folder SearchINH_IPSP --params INH_ipsp_response.py --search search.py --map yes nest

# python plot_map.py
