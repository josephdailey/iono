from cddis_access import *
import os
import sys
from datetime import date

today = date.today()
email = sys.argv[1]

if len(sys.argv) > 2:
    min_year = int(sys.argv[2])
else:
    min_year = 1999
assert min_year >= 1993

if len(sys.argv) > 3:
    max_year = int(sys.argv[3])
else:
    max_year = int(today.strftime("%Y"))
assert max_year <= int(today.strftime("%Y"))

session = cddis_login(email)

os.chdir("data")
for year in range(min_year, max_year + 1):
    session.cwd("/gnss/products/ionex/%s" % year)
    days = session.nlst()
    for day in days:
        session.cwd(day)
        files = session.search_ionex_old()
        session.dl_list(files)
        files = session.search_ionex_new()
        session.dl_list(files)
        session.cwd("..")
os.chdir("..")

session.quit()