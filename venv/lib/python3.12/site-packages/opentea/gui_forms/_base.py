from __future__ import (
    annotations,
)  # Needed to use same type as the enclosing class in typing
import abc

from opentea.gui_forms._exceptions import GetException
from loguru import logger

import inspect
from opentea.gui_forms.monitors import add_monitor

class OTTreeElement(metaclass=abc.ABCMeta):
    """The common aspect btw nodeWidgets and LeafWidgets

    No Tk in here
    """

    def __init__(self, schema: dict, parent: OTTreeElement, name: dict):
        self.parent = parent
        self.name = name

        # Schema information -------------------
        self.schema = schema
        self.title = self.schema.get("title", f"#{self.name}")
        self.properties = self.schema.get("properties", [])
        self.item_type = self.schema.get("type", [])
        self.state = self.schema.get("state", "normal")  # either "normal" of "disabled"
        if self.item_type == "array":
            self.item_type = schema["items"]["type"]
        # Schema information -------------------

        self.kind = None  # Either "leaf" or "node"
        self.children = dict()
        # Contains
        # {
        #     child_name : child_node,
        #     child_name2 : child_node2,
        #     child_name3 : child_node3,
        # }

        # When you create the element, it adds itself to its parent familly
        self.parent.add_child(self)
        self.my_root_tab_widget = parent.my_root_tab_widget
        # This must precede check_if_dependent() which require to know the masterTab

        # Dependencies elements (flags ot_require)
        self.slaves = []
        self.master = None
        self.dependent = self.check_if_dependent()

        # THE STATUS
        self._status = 0
        self._process_status = None  # used fot TabNodes evaluation

    @abc.abstractmethod
    def get(self):
        """Get the information from a widget"""
        pass

    @abc.abstractmethod
    def set(self, value):
        """Set the information to a widget"""
        pass

    def add_child(self, child):
        """How to add a children to the element

        CaveAt : one could thik this is limited to nodes, but here the catch:
        present here because complex leaves like lists can have child widgets
        """
        self.children[child.name] = child

    def list_children(self):
        """How to how to iterate over the list of children
        """
        return self.children.values()

    ####################
    # Status handling

    # def update_status(self, changing=False, descending=False):
    #     """Introspection to update status"""
    #     # on change, for everyone
    #     # leaves does not use change because of TvVar tracking onvar_change
    #     add_monitor("update_status")
    #     if self.kind == "leaf":
    #         try:
    #             value = self.get()
    #             if self.leaf_is_valid(value):
    #                 if changing:
    #                     status = 0
    #                     if self.previous_value == value:
    #                         status = 1
    #                 else:
    #                     status = 1
    #             else:
    #                 status = -1
    #         except GetException:
    #             status = -1
    #         if status == 1:
    #             self.previous_value = value

    #     elif self.kind == "dynlist_leaf":
    #         if changing:
    #             status = 0
    #         else:
    #             status = 1
    #             for child in self.list_children():
    #                 status = min(child._status, status)
    #             if status == 1:
    #                 self.previous_value = self.get()

    #     elif self.kind == "node":
    #         if changing:
    #             status = 0
    #         else:
    #             status = 1
    #             for child in self.list_children():
    #                 status = min(child._status, status)
    #             if (
    #                 self._process_status is not None
    #             ):  # used for tab widgets, to take into accout the process success
    #                 status = min(self._process_status, status)

    #     elif self.kind in [
    #         "dead_leaf",
    #         "unpacked_node",
    #     ]:  # when leaf is just for documentation, info, etc..
    #         self._status = 1
    #         self.once_validated()
    #         self.refresh_status_display()
    #         #self.update_status_predecessors()
    #         return
    #     else:
    #         self.ping(stack=True)
    #         raise RuntimeError("How did you get there?")

    #     if status == 1:
    #         self.once_validated()

    #     self._status = status
    #     self.refresh_status_display()
    #     if descending :
    #         if self.children == {}:
    #             self.update_status_predecessors()
    #     else:
    #         self.update_status_predecessors()

    # def update_status_predecessors(self, changing: bool = False):
    #     """RECURSIVE to refresh all successors"""
    #     self.parent.update_status(changing=changing)
    #     self.parent.update_status_predecessors(changing=changing)

    # def update_status_successors(self):
    #     """RECURSIVE to refresh all successors"""
    #     for child in self.list_children():
    #         child.update_status(descending=True)
    #         child.update_status_successors()


    #new version of updates statuses
    def evaluate_local_status(self, changing=False):
        """Evaluate local status
        
        - Evaluation only for the leaves, never for the nodes
        - no graphical updates
        - never call for parents or childs
        """
        # on change, for everyone
        # leaves does not use change because of TvVar tracking onvar_change
        if changing:
            add_monitor("evaluate_local_status_changing")
        else:
            add_monitor("evaluate_local_status_validate")
    
        if self.kind == "leaf":
            try:
                value = self.get()
                if self.leaf_is_valid(value):
                    if changing:
                        status = 0
                        if self.previous_value == value:
                            status = 1
                    else:
                        status = 1
                else:
                    status = -1
            except GetException:
                status = -1
            if status == 1:
                self.previous_value = value

        elif self.kind == "dynlist_leaf":
            if changing:
                status = 0
            else:
                status = 1
                for child in self.list_children():
                    status = min(child._status, status)
                if status == 1:
                    self.previous_value = self.get()

        elif self.kind == "node":
            if changing:
                status = 0
            else:
                status = 1
                for child in self.list_children():
                    status = min(child._status, status)
                if (
                    self._process_status is not None
                ):  # used for tab widgets, to take into accout the process success
                    status = min(self._process_status, status)

        
        elif self.kind in [
            "dead_leaf",
            "unpacked_node",
        ]:  # when leaf is just for documentation, info, etc..
            status = 1
        else:
            self.ping(stack=True)
            raise RuntimeError("How did you get there?")

        if status == 1:
            self.once_validated()

        self._status = status
       

    def evaluate_status_descending(self,changing:bool=False):
        """This is for the status update of the gui
        
        - evaluation  of statuses, but no agregation
        - NO GRAPHICAL UPDATE
        """
        for child in self.list_children():
            child.evaluate_status_descending(changing=changing)

        self.evaluate_local_status(changing=changing)

    def refresh_status_display_descending(self):
        """This is for the graphical update of the GUI
        
        - Graphical update, ONLY THERE
        """
        for child in self.list_children():
            child.refresh_status_display_descending()

        self.refresh_status_display()


    def evaluate_status_ascending(self,changing:bool=False):
        """This is for the status update of the gui
        
        - evaluation  of statuses, but no agregation
        - NO GRAPHICAL UPDATE
        """
        self.evaluate_local_status(changing=changing)
        try:
            self.parent.evaluate_status_ascending()
        except AttributeError: # stop at the top
            pass
        
    def refresh_status_display_ascending(self):
        """This is for the graphical update of the GUI
        
        - Graphical update, ONLY THERE
        """        
        self.refresh_status_display()
        try:
            self.parent.refresh_status_display_ascending()
        except AttributeError: # stop at the top
            pass



    def refresh_status_display(self):
        """Additional operations to perform when updating status.

        Redefined in XOR, Multiples , and Tabs,
        Redefined in Leaves
        """
        pass

    def once_validated(self):
        """What to do after successful validation"""
        pass

    def leaf_is_valid(self, value=None) -> bool:
        """How to check a leaf is valid"""
        return True

    #    Validation: when the user test the current input
    #####################################################

    #####################################################
    #    MASTER - SLAVE Dependencies (ot_require)
    def check_if_dependent(self):
        """Configure for a dependency. Element is the slave in this case"""
        if "ot_require" not in self.schema:
            return False
        master_name = self.schema.get("ot_require")
        self.my_root_tab_widget.add_dependency(master_name, self)
        return True

    def add_dependents(self, slaves):
        """
        Add all slaves , PLUS set data to these slaves
        """
        try:
            data = self.get()
        except GetException:
            data = None

        for slave in slaves:
            self._add_dependent(slave, data)

    def _add_dependent(self, slave, data=None):
        """
        Addition of one dependency


        Add object slave to list of dependent slaves
        state to object who the master is
        ask for a set() of the slave according to data
        """
        if slave not in self.slaves:
            self.slaves.append(slave)
            slave.master = self

        if data is not None:
            slave.set_slaves(data)

    def set_slaves(self, value):
        """Trigger a set() in each slave"""
        for slave in self.slaves:
            slave.slave_update(value)  # make a specific method for  slave update
            pass

    def _reset_master(self):
        """Used in case of destroy() method"""
        if self.master is None:
            return
        self.master.slaves.remove(self)
        self.master = None

    #    MASTER - SLAVE Dependencies (ot_require)
    #####################################################

    #####################################################
    # Utilities
    #

    def ping(self, stack=False):
        """ "Just add logging element, for debug"""
        logger.warning(
            f"PING :{self.ottype}({self.kind})|{self.name}|{self.title} status:{self._status}"
        )
        if stack:
            call_stack_str(2)

    @property
    def ottype(self) -> int:
        """Return Opentea  Object type

        Used for debugging or fancy viz.
        """
        return str(type(self)).split(".")[-1].split("'")[0]

    def get_child_by_name(self, name: str) -> OTTreeElement:
        """Recursive method to find a child in the tree

        Used for the declaration of dependencies in otroot.
        """
        # check if child is at this level
        for child in self.children.values():
            if child.name == name:
                return child

        # check children
        for child in self.children.values():
            child_ = child.get_child_by_name(name)

            if child_ is not None:
                return child_

        return None


##############################
# Debug helpers
def call_stack_str(skip_first: int = 1):
    """print the call stack. Useful for a recursive structure"""
    names = []
    for ob in inspect.stack()[skip_first:]:
        filen = ob.filename.split("/")[-1]
        names.append(f"{ob.function} {ob.lineno}:{filen}")
        if ob.function == "__call__":
            break

    indent = 0
    str_ = "\n"
    for name in names:
        str_ += " " * indent + "|" + name + "\n"
        indent += 2
    return str_


#
######################################
