"""Create the synthetic CF-NetCDF dataset used by the other examples.

Run:  python examples/generate_data.py
"""

from pathlib import Path

from cwa.data import write_netcdf

if __name__ == "__main__":
    out = Path("cwa_demo.nc")
    # chunk_time=1 stores the file so a single time slice can be read without
    # decompressing the whole variable -- matches the streaming read pattern.
    write_netcdf(out, n_time=240, n_lat=90, n_lon=180, chunk_time=1)
    print(f"wrote {out.resolve()}  ({out.stat().st_size / 1e6:.1f} MB)")
