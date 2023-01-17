HEURES_CREUSES = ("00:00", "07:00")


class Schedule:

    def __init__(self, logement_name, parent_name):
        self.logement_name = logement_name
        self.parent_name = parent_name
        self.cost = 1.0

    def get_cost(self):
        return self.cost

    def get_parent_name(self):
        return self.parent_name


s = Schedule("2", '5')
print(s.get_cost())

# %%
