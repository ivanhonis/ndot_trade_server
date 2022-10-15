import os
import shutil

import numpy as np
import configparser
import time
import datetime
import pickle
import psutil


class NTradeFolderManager:
    def __init__(self):
        self.main_dir = os.getcwd()
        self.slots_dir = self.main_dir + r'/SLOTS'
        self.clients_dir = self.main_dir + r'/CLIENTS'
        self.config_dir = self.main_dir + r'/CONFIGS'
        self.messages_dir = self.main_dir + r'/MESSAGES'
        self.fn_trade_rights = self.main_dir + r'/CONFIGS/nDot_trade_rights.pickle'
        self.clients_rights = {}
        self.trade_rights = {}
        self.commands = {"trade": "nDot_command_trade.pickle",
                         "push": "nDot_command_push.npy"}

    @staticmethod
    def log(text, line=False):
        if line:
            print('-' * 90)
        print(datetime.datetime.now(), " - ", text)

    def add_slot(self, project):
        newpath = self.slots_dir
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        newpath = self.slots_dir + r"/" + project
        if not os.path.exists(newpath):
            self.log(f"add_slot: {project}")
            os.makedirs(newpath)

    def remove_slot(self, project):
        delpath = self.slots_dir + r"/" + project
        if os.path.exists(delpath):
            self.log(f"remove_slot: {project}")
            shutil.rmtree(delpath)

    def add_client(self, client):

        newpath = self.clients_dir
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        newpath = self.clients_dir + r"/" + client
        if not os.path.exists(newpath):
            self.log(f"add_client: {client}")
            os.makedirs(newpath)

        newpath = self.clients_dir + r"/" + client + r"/INCOMING"
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        newpath = self.clients_dir + r"/" + client + r"/INCOMING/ATTACHMENT"
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        newpath = self.clients_dir + r"/" + client + r"/OUTGOING"
        if not os.path.exists(newpath):
            os.makedirs(newpath)

    def remove_client(self, client):
        delpath = self.clients_dir + r"/" + client
        if os.path.exists(delpath):
            self.log(f"remove_client: {client}")
            shutil.rmtree(delpath)

    def get_client_incoming_commands(self, client):
        path = self.clients_dir + r"/" + client + r"/INCOMING"
        return os.listdir(path)

    def remove_client_incoming(self, client, dfname):
        fname = self.clients_dir + r"/" + client + r"/INCOMING/" + dfname
        if os.path.exists(fname):
            os.remove(fname)

    def set_trade(self, client):
        fn_from = self.clients_dir + r"/" + client + r"/INCOMING/" + self.commands['trade']
        fn_to = self.fn_trade_rights
        if os.path.exists(fn_from):
            self.log(f"set_trade: {client}")
            shutil.move(fn_from, fn_to)
            self.get_status()

    def get_trade_rights(self):
        if os.path.exists(self.fn_trade_rights):
            return pickle.load(open(self.fn_trade_rights, "rb"))
        else:
            return {}

    def get_trade_right_by_project(self, project):
        if os.path.exists(self.fn_trade_rights):
            self.trade_rights = self.get_trade_rights()
            if project in self.trade_rights:
                return self.trade_rights[project]
            else:
                return False
        else:
            return False

    def set_model(self, client, project):
        tf_name = f"nDot_TF_MODEL_{project}.h5"
        minmax_name = f"nDot_MinMaxScaler_{project}.pickle"
        config_name = f"nDot_PRO_{project}.txt"

        fn_project_to = self.slots_dir + r"/" + project

        fn_from = self.clients_dir + r"/" + client + r"/INCOMING/ATTACHMENT/" + tf_name
        fn_to = self.slots_dir + r"/" + project + r"/" + tf_name
        if os.path.exists(fn_from) and os.path.exists(fn_project_to):
            self.log(f"set TF model: {client} -> {project}")
            shutil.move(fn_from, fn_to)
        elif os.path.exists(fn_from):
            os.remove(fn_from)

        fn_from = self.clients_dir + r"/" + client + r"/INCOMING/ATTACHMENT/" + minmax_name
        fn_to = self.slots_dir + r"/" + project + r"/" + minmax_name
        if os.path.exists(fn_from) and os.path.exists(fn_project_to):
            self.log(f"set MinMax: {client} -> {project}")
            shutil.move(fn_from, fn_to)
        elif os.path.exists(fn_from):
            os.remove(fn_from)

        fn_from = self.clients_dir + r"/" + client + r"/INCOMING/ATTACHMENT/" + config_name
        fn_to = self.slots_dir + r"/" + project + r"/" + config_name
        if os.path.exists(fn_from) and os.path.exists(fn_project_to):
            self.log(f"set project config: {client} -> {project}")
            shutil.move(fn_from, fn_to)
        elif os.path.exists(fn_from):
            os.remove(fn_from)

    # command structure ---------------------------------------------
    #     trade = {"BTCUSDT_P10INTX": True,
    #              "ETHUSDT_P10INTX": True}
    #
    #     push = ["BTCUSDT_P10INTX",
    #             "ETHUSDT_P10INTX"]

    def process_client_incoming(self, client):
        cmd_list = self.get_client_incoming_commands(client)
        if len(cmd_list) > 0:
            for cmd in cmd_list:
                if cmd == self.commands['trade'] and self.clients_rights[client]['right_command_trade']:
                    self.log(f"command: trade {client}")
                    self.set_trade(client)
                elif cmd == self.commands['push'] and self.clients_rights[client]['right_command_push']:
                    self.log(f"command: push {client}")
                    self.process_slots(client)
                if cmd != "ATTACHMENT":
                    self.remove_client_incoming(client, cmd)
                    self.log(f"{cmd} - removed")

    def process_slots(self, client):
        fn_push = f"{self.clients_dir}/{client}/INCOMING/{self.commands['push']}"
        ipush = np.load(fn_push)
        for project in ipush:
            self.add_slot(project)
            self.set_model(client, project)
        self.remove_off_slots(ipush)

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

    def remove_off(self):
        #TODO: remove_off

        # mindent eltakarít ami nem kell
        # ezek hibás másolások ereményei lehetnek
        # olyan clientnél akinek nincs push joga nem lehet semmi a attacmentben
        # client alatt csak incomeing és outgoing lehet mindent mást ki kell törölni
        # slot alatt csak alkönyvátrak lehetnek minden mást ki kell törölni
        # slotban csak model minmax és projet lehet
        pass

    def remove_off_clients(self, clients):
        path = self.clients_dir
        dlist = os.listdir(path)
        diff_list = list(set(dlist) - set(clients))
        for dfl in diff_list:
            self.remove_client(dfl)

    def remove_off_slots(self, ipush):
        path = self.slots_dir
        dlist = os.listdir(path)
        diff_list = list(set(dlist) - set(ipush))
        for project in diff_list:
            self.remove_slot(project)

    def process_clients(self):
        config = configparser.ConfigParser()
        conf_fname = self.config_dir + r"/nDot_clients.ini"
        config.read(conf_fname)
        self.set_clients_rights(config)
        clients = config.sections()
        for cl in clients:
            self.add_client(cl)
            self.process_client_incoming(cl)
        self.remove_off_clients(clients)
        self.remove_off()

    def messages_to_clients(self):
        messages = os.listdir(self.messages_dir)
        clients = os.listdir(self.clients_dir)
        for me in messages:
            from_fn = self.messages_dir + r"/" + me
            for cl in clients:
                to_fn = self.clients_dir + r"/" + cl + r"/OUTGOING/" + me
                shutil.copyfile(from_fn, to_fn)

        for me in messages:
            from_fn = self.messages_dir + r"/" + me
            os.remove(from_fn)

    def get_status(self):
        clients = os.listdir(self.clients_dir)
        if os.path.exists(self.slots_dir):
            slots = os.listdir(self.slots_dir)
        else:
            slots = {}
        models = {}
        for project in slots:
            models[project] = {'tf': 0.0,
                               'minmax': 0.0,
                               'config': 0.0}
            tf_name = self.slots_dir + f"/{project}/nDot_TF_MODEL_{project}.h5"
            minmax_name = self.slots_dir + f"/{project}/nDot_MinMaxScaler_{project}.pickle"
            config_name = self.slots_dir + f"/{project}/nDot_PRO_{project}.txt"
            if os.path.exists(tf_name):
                models[project]["tf"] = os.path.getmtime(tf_name)
            if os.path.exists(minmax_name):
                models[project]["minmax"] = os.path.getmtime(minmax_name)
            if os.path.exists(config_name):
                models[project]["config"] = os.path.getmtime(config_name)

        vm = psutil.virtual_memory()
        vmem = {'total': int(vm.total),
        'available': int(vm.available),
        'percent': float(vm.percent)}

        du = psutil.disk_usage("/")
        dusage = {'total': int(du.total),
        'used': int(du.used),
        'free': int(du.free),
        'percent': float(du.percent)}

        proc = list(psutil.cpu_percent(interval=1, percpu=True))

        status = {"datetime": datetime.datetime.now(),
                  "clients": clients,
                  "slots": models,
                  "trade_rights": self.get_trade_rights(),
                  "processors": proc,
                  "virtual_memory": vmem,
                  "disk_usage": dusage}

        fname = self.messages_dir + r"/nDot_trade_server_status.pickle"
        pickle.dump(status, open(fname, "wb"))

        self.log(str(status["trade_rights"]), line=True)
        self.log(str(status["slots"]))
        self.log(str(status["clients"]))

if __name__ == "__main__":
    # print(psutil.cpu_percent(interval=.25, percpu=True))

    tr = NTradeFolderManager()

    # tr.add_slot("BTCUSDT1")
    # tr.remove_slot("BTCUSDT2")
    # tr.add_client("X")
    # # tr.remove_client("X2")
    # print(tr.get_client_incoming("X"))
    #
    # tr.process_client_incoming("X")

    # tr.remove_client_incoming("x", "y")
    push = np.array(["BTCUSDT_P10INT",
          "ETHUSDT_P10INT"])

    np.save("nDot_command_push", push)

    # trade = {"BTCUSDT_P10INTX": True,
    #              "ETHUSDT_P10INTX": True}
    #
    # pickle.dump(trade, open("nDot_command_trade.pickle", "wb"))

    while True:
        tr.process_clients()
        tr.get_status()
        tr.messages_to_clients()

        time.sleep(25)
