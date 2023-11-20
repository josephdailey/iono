from ftplib import FTP_TLS
from ssl import SSLContext
import re
import os
from typing import List

class cddis_session(FTP_TLS):
    def __init__(self, host: str = "", user: str = "", passwd: str = "", acct: str = "", keyfile: str | None = None, certfile: str | None = None, context: SSLContext | None = None, timeout: float = ..., source_address: tuple[str, int] | None = None, *, encoding: str = "utf-8") -> None:
        super().__init__(host, user, passwd, acct, keyfile, certfile, context, timeout, source_address, encoding=encoding)

    def search_ionex_new(self, SRC:str = "IGS", TYP:str = "FIN", SMP:str = "02H", CNT:str = "GIM") -> list:
        regex = re.compile("%s0OPS%s_\d{11}_01D_%s_%s\.INX\.gz" % (SRC, TYP, SMP, CNT))
        return list(filter(regex.match, self.nlst()))
                
    def search_ionex_old(self, SRC:str = "igs") -> list:
        regex = re.compile("%sg\d{3}\d\.\d{2}i\.Z" % SRC)
        return list(filter(regex.match, self.nlst()))
    
    def dl_list(self, filenames: List[str]):
        for filename in filenames:
            self.retrbinary("RETR " + filename, open(filename, 'wb').write)
            if filename.endswith(".gz") or filename.endswith(".Z"):
                os.system("gzip -df %s" % filename)

def cddis_login(email:str) -> cddis_session:
    session = FTP_TLS(host = "gdc.cddis.eosdis.nasa.gov")
    session.login(user="anonymous", passwd=email)
    session.prot_p()

    #Slightly hacky but this works SO LONG AS only methods are added!
    session.__class__ = cddis_session
    return session