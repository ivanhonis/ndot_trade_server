import os
import shutil
import psutil
import configparser

class ntrade_folder_manager:
    def __init__(self):
        self.slots_dir = r'.\\SLOTS'
        self.clients_dir = r'.\\CLIENTS'
        self.clients_rights = {}
        self.status = {}
        self.messages = []

    def log(self, text):
        print(text)

    def add_slot(self, project):
        newpath = self.slots_dir
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        newpath = self.slots_dir + "\\" + project
        if not os.path.exists(newpath):
            os.makedirs(newpath)

    def remove_slot(self, projekt):
        delpath = self.slots_dir + "\\" + projekt
        if os.path.exists(delpath):
            shutil.rmtree(delpath)

    def add_client(self, client):

        newpath = self.clients_dir
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        newpath = self.clients_dir + "\\" + client
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        newpath = self.clients_dir + "\\" + client + "\\INCOMING"
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        newpath = self.clients_dir + "\\" + client + "\\OUTGOING"
        if not os.path.exists(newpath):
            os.makedirs(newpath)

    def remove_client(self, client):
        delpath = self.clients_dir + "\\" + client
        if os.path.exists(delpath):
            shutil.rmtree(delpath)

    def get_client_incoming(self, client):
        path = self.clients_dir + "\\" + client + "\\INCOMING"
        return os.listdir(path)

    def remove_client_incoming(self, client, dfname):
        fname = self.clients_dir + "\\" + client + "\\INCOMING\\" + dfname
        if os.path.exists(fname):
            os.remove(fname)

    def process_client_incoming(self, client):
        commands = self.get_client_incoming(client)
        if len(commands) > 0:
            for cmd in commands:
                if cmd == "nDot_command_slots.npy" and self.clients_rights[client]['right_command_slot']:
                    pass
                    self.remove_client_incoming(client, cmd)
                elif cmd == "nDot_command_trade.npy" and self.clients_rights[client]['right_command_trade']:
                    pass
                    self.remove_client_incoming(client, cmd)
                elif cmd == "nDot_command_push.npy" and self.clients_rights[client]['right_command_push']:
                    pass
                    # át kell tenni a megérkezett fileokat
                    self.remove_client_incoming(client, cmd)

    def set_clients_rights(self, config):

        def bl(istr):
            if istr == "True":
                return True
            else:
                return False

        self.clients_rights = {}
        for cl in config.sections():
            self.clients_rights[cl] = {}
            for rh in config.options(cl):
                self.clients_rights[cl][rh] = bl(config.get(cl, rh))

    def remove_off_clients(self, clients):
        path = self.clients_dir
        print(os.listdir(path))

    def process_clients(self):
        config = configparser.ConfigParser()
        conf_fname = self.clients_dir + "\\" + "nDot_clients.ini"
        config.read(conf_fname)
        self.set_clients_rights(config)
        clients = config.sections()
        for cl in clients:
            self.add_client(cl)
            self.process_client_incoming(cl)
    # print(config.options('SectionOne'))
    # print(config.get('SectionOne', 'Status'))


if __name__ == "__main__":
    # print(psutil.cpu_percent(interval=.25, percpu=True))

    tr = ntrade_folder_manager()

    # tr.add_slot("BTCUSDT1")
    # tr.remove_slot("BTCUSDT2")
    # tr.add_client("X")
    # # tr.remove_client("X2")
    # print(tr.get_client_incoming("X"))
    #
    # tr.process_client_incoming("X")

    # tr.remove_client_incoming("x", "y")
    tr.process_clients()
