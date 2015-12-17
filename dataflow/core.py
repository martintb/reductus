"""
Core class definitions
"""
import sys
from collections import deque
import inspect
import json
import re

from .deps import processing_order

TEMPLATE_VERSION = '1.0'

_instrument_registry = []
_module_registry = {}
_data_registry = {}
def register_instrument(instrument):
    """
    Add a new instrument to the server.
    """
    _instrument_registry.append(instrument.id)
    for m in instrument.modules:
        register_module(m)
    for d in instrument.datatypes:
        register_datatype(d)

def register_module(module):
    """
    Register a new calculation module.
    """
    if module.id in _module_registry and module != _module_registry[module.id]:
        #raise TypeError("Module already registered")
        return
    _module_registry[module.id] = module
    
def lookup_module(id):
    """
    Lookup a module in the registry.
    """
    return _module_registry[id]

def register_datatype(datatype):
    if datatype.id in _data_registry and datatype != _data_registry[datatype.id]:
        raise TypeError("Datatype already registered")
    _data_registry[datatype.id] = datatype

def lookup_datatype(id):
    return _data_registry[id]

class Module(object):
    """
    Processing module

    A computation is represented as a set of modules connected by wires.

    *id* : string
        Module identifier. By convention this will be a dotted structure
        '<instrument class>.<operation>.<qualifier>', with qualifier optional.
        For example, use "tas.load" for triple axis data and "sans.load" for
        SANS data.  For NCNR SANS data format loaders, use "sans.load.ncnr".

    *version* : string
        Version number of the code which implements the filter calculation.
        If any code in the supporting libraries changes in a way that will
        affect the calculation results, the version number should be
        incremented.  This includes bug fixes.

    *name* : string
        The display name of the module. This may appear in the user interface
        in addition to any pictorial representation of the module. Usually it
        is just the name of the operation. By convention, it should have
        every word capitalized, with spaces between words.

    *description* : string
        A tooltip shown when hovering over the icon

    *icon* : { URI: string, terminals: { *id*: [*x*,*y*,*i*,*j*] } }
        Image representing the module, or none if the module should be
        represented by name.

        The terminal locations are identified by:

            *id* : string
                name of the terminal

            *position* : [int, int]
                (x,y) location of terminal within icon

            *direction* : [int, int]
                direction of the wire as it first leaves the terminal;
                default is straight out

    *fields* : [Parameter]
        An form defining the constants needed for the module. For
        example, an attenuator will have an attenuation scalar. Field
        names must be distinct from terminal names.

        *id* : string
            name of the variable associated with the date

        *label* : string
            display name for the terminal.



    *terminals* : [Terminal]
        List module inputs and outputs.

        *id* : string
            name of the variable associated with the data; this must
            correspond to a parameter name in the module action.

        *label* : string
            display name for the terminal.

        *type* : string
            name of the datatype associated with the data, with the
            output of one module needing to match the input of the
            next. Using a hierarchical type system, such as
            refl.data, we can attach to generic modules like scaling
            as well as specific modules like footprint correction. By
            defining the type of data that flows through the terminal
            we can highlight valid connections automatically.

        *use* : string | "in|out"
            whether this is an input parameter or an output parameter

        *description* : string
            A tooltip shown when hovering over the terminal; defaults
            to datatype name

        *required* : boolean
            true if an input is required; ignored on output terminals.

        *multiple* : boolean
            true if multiple inputs are accepted; ignored on output
            terminals.

        *action* : callable
            function which performs the action

        *module* : string
            name of the module which contains the action.  You should
            be able to import id from module to get action.
    """
    def __init__(self, id, version, name, description, icon=None,
                 terminals=None, fields=None, action=None,
                 author="", module="",
                 xtype=None, filterModule=None,
                 ):
        self.id = id
        self.version = version
        self.name = name
        self.description = description
        self.icon = icon
        self.fields = fields if fields is not None else {}
        self.terminals = terminals
        self.action = action
        self.module = module
        self.xtype = xtype
        self.filterModule = filterModule

    def get_terminal_by_id(self, id):
        """ 
        Lookup terminal by id, and return.
        Returns None if id does not exist.
        """
        terminal_lookup = dict((t['id'], t) for t in self.terminals)
        return terminal_lookup[id]
        
    def get_source_code(self):
        """
        Retrieves the source code for the identified module that
        does the actual calculation.  If no module is identified
        it returns an empty string
        """
        source = ""
        if self.filterModule is not None:
            source = "".join(inspect.getsourcelines(self.filterModule)[0])
        return source        
        
class Template(object):
    """
    A template captures the computational workflow as a wiring diagram.

    *name* : string
        String identifier for the template

    *description* : string
        Extended description to be displayed as help to the template user.

    *modules* : [TemplateModule]
        Modules used in the template

        *module* : string
            module id for template node

        *version* : string
            version number of the module

        *config* : { field: value }
            initial values for the fields

        *position* : [int,int]
            location of the module on the canvas.

    *wires* : [TemplateWire]
        Wires connecting the modules

        *source* : [int, string]
            module id in template and terminal name in module

        *target* : [int, string]
            module id in template and terminal name in module

    *instrument* : string
        Instrument to which the template applies

    *version* : string
        Template version number
    """
    def __init__(self, name=None, description=None, modules=None, wires=None,
                 instrument="", version=TEMPLATE_VERSION):
        self.name = name
        self.description = description
        self.modules = modules
        self.wires = wires
        self.instrument = instrument
        self.version = version

    def order(self):
        """
        Return the module ids in processing order.
        """
        pairs = [(w['source'][0], w['target'][0]) for w in self.wires]
        return processing_order(len(self.modules), pairs)

    def __iter__(self):
        """
        Yields module#, inputs for each module in the template in order.
        """
        for id in self.order():
            inputs = [w for w in self.wires if w['target'][0] == id]
            yield id, inputs

    def __getstate__(self):
        """
        Version aware pickler. Returns (version, state)
        """
        return TEMPLATE_VERSION, self.__dict__
    def __setstate__(self, state):
        """
        Version aware unpickler. Expects (version, state)
        """
        version, state = state
        if version != TEMPLATE_VERSION:
            raise TypeError('Template definition mismatch')
        self.__dict__ = state
        
    def get_parents(self, id):
        """
        Retrieve the data objects that go into the inputs of a module
        """
        parents = [w for w in self.wires if w['target'][0] == id]
        return parents


class Instrument(object):
    """
    An instrument is a set of modules and standard templates to be used
    for reduction

    *id* : string
        Instrument identifier. By convention this will be a dotted
        structure '<facility>.<instrument class>.<instrument>'

    *name* : string
        The display name of the instrument

    *menu* : [(string, [Module, ...]), ...]
        Modules available. Modules are organized into groups of related
        operations, such as Input, Reduce, Analyze, ...

    *datatypes* : [Datatype]
        List of datatypes used by the instrument

    *archive* : URI
        Location of the data archive for the instrument. Archives must
        implement an interface that allows data sets to be listed and
        retrieved for a particular instrument/experiment.
    """
    def __init__(self, id, name=None, menu=None,
                 datatypes=None, requires=None, archive=None, loaders=None):
        self.id = id
        self.name = name
        self.menu = menu
        self.datatypes = datatypes
        self.requires = requires
        self.archive = archive
        self.loaders = loaders

        self.modules = []
        for _, m in menu:
            self.modules.extend(m)
        self._check_datatypes()
        self._check_names()

    def _check_datatypes(self):
        defined = set(d.id for d in self.datatypes)
        used = set()
        for m in self.modules:
            used |= set(t['datatype'] for t in m.terminals)
        if used - defined:
            raise TypeError("undefined types: %s" % ", ".join(used - defined))
        if defined - used:
            raise TypeError("unused types: %s" % ", ".join(defined - used))

    def _check_names(self):
        names = set(m.name for m in self.modules)
        if len(names) != len(self.modules):
            raise TypeError("names must be unique within an instrument")
        
    def id_by_name(self, name):
        for m in self.modules:
            if m.name == name: return m.id
        raise KeyError(name + ' does not exist in instrument ' + self.name)
        
class Data(object):
    """
    Data objects represent the information flowing over a wire.

    *name* : string
        User visible identifier for the data. Usually this is file name.

    *datatype* : string
        Type of the data. This determines how the data may be plotted
        and filtered.

    *intent* : string
        What role the data is intended for, such as 'background' for
        data that is used for background subtraction.

    *dataid* : string
        Key to the data. The data itself can be stored and retrieved by key.

    *history* : list
        History is the set of modules used to create the data. Each module
        is identified by the module id, its version number and the module
        configuration used for this data set. For input terminals, the
        configuration will be {string: [int,...]} identifying
        the connection between nodes in the history list for each input.

    *module* : string

    *version* : string

    *inputs* : { <input terminal name> : [(<hist iindex>, <output terminal>), ...] }

    *config* : { <field name> : value, ... }

    *dataid* : string
    """
    def __new__(subtype, id, cls, loaders=[]):
        obj = object.__new__(subtype)
        obj.id = id
        obj.cls = cls
        obj.loaders = loaders
        return obj
    
    def __getstate__(self):
        return "1.0", self.__dict__
    
    def __setstate__(self, state):
        version, state = state
        self.__dict__ = state
        
    def get_plottable(self):
        return self.dumps()
    
    def dumps(self):
        return json.dumps(self.__dict__)
    
    @classmethod
    def loads(cls, str):
        return Data(str, Data)


# ============= Parent traversal =============
class Node(object):
    """
    Base node

    A diagram is created by connecting nodes with wires.

    *parents* : [Node]
        List of parents that this node has

    *params* : dictionary
        Somehow matches a parameter to its current value,
        which will be compared to the previous value as found
        in the database.
    """
    def __init__(self, parents, params):
        self.parents = parents
        self.params = params
        
    def searchDirty(self):
        queue = deque([self])
        while queue:
            node = queue.popleft()
            if node.isDirty():
                return True
            for parent in node.parents:
                queue.append(parent)
        return False
        
    def isDirty(self):
        # Use inspect or __code__ for introspection?
        return self.params != self._get_inputs()
    
    def _get_inputs(self):
        # Get data from database
        #pass
        data = {'maternal grandpa':{'id':'maternal grandpa'},
                'maternal grandma':{'id':'maternal grandma'},
                'mom':{'id':'mom'},
                'paternal grandpa':{'id':'paternal grandpa'},
                'paternal grandma':{'id':'paternal grandma'},
                'dad':{'id':'dad'},
                'son':{'id':'son'}, }
        return data.get(self.params['id'], {})
    
if __name__ == '__main__':
    head = Node([Node([Node([], {'id':'maternal grandpa'}),
                       Node([], {'id':'maternal grandma'})],
                      {'id':'mom'}),
                 Node([Node([], {'id':'paternal grandpa'}),
                       Node([], {'id':'paternal grandma'})],
                      {'id':'dad'})],
                {'id':'son'})
    print "Dirty" if head.searchDirty() else "Clean"


def bundle(fn):
    """
    Decorator which turns a single file function into a bundle function.

    Note: untested code.

    Note: not sure we want to do this; it would be better for the
    infrastructure to identify that a bundle is given to an action
    and unless the action is known to accept a bundle, it will open
    up a table of parameter values, one for each file in the bundle
    so that the can be set individually.
    """
    argspec = inspect.getargspec(fn)
    full_args = argspec.args
    call_args = full_args if not inspect.ismethod(fn) else full_args[1:]
    first = call_args[0]
    sig = ",".join(full_args)
    call = ",".join(call_args)
    source = """
    def wrapper(%(sig)s):
        if isinstance(%(first)s, list):
            return [fn(%(call)s) for d in %(first)s]
        else:
            return fn(%(call)s)
    """%dict(first=first, sig=sig, call=call)
    context = dict(fn=fn)
    code = compile(source, fn.__file__, 'single')
    exec(code, context)
    wrapper = context['wrapper']
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    wrapper.__module__ = fn.__module__
    wrapper.__defaults__ = argspec.defaults
    wrapper.__annotations__ = fn.__annotations__
    wrapper.__dict__.update(fn.__dict__)


def auto_module(action):
    """
    Given an action function, parse the docstring and return a node.

    The action name and docstring are highly stylized.

    The description is first.  It can span multiple lines.

    The parameter sections are ``**Inputs**``, ``**Outputs**`` and
    ``**Fields**``.  These must occur on a line by themselves, with
    the ``**...**`` markup to make them show up bold in the sphinx docs.

    Each parameter has name, type, description and optional default value.
    The name is the first on the line, followed by the type in
    (parentheses), then ':', a description string and default value in
    [brackets].  The parameter definition can span multiple lines, but
    the description will be joined to a single line.

    If the input is optional, then mark the type with '?'.  If the node
    accepts zero or more inputs, then mark the type with '*'.  If the node
    accepts one or more inputs, then mark the type with '+'.

    The possible types are determined by the

        str, int, float, [float], opt1|opt2|...|optn


    The resulting doc string should look okay in sphinx. For example,

    ``
        Fit detector dead time constants (paralyzing and non-paralyzing) from
        measurement of attenuated and unattenuated data.

        **Inputs**

        attenuated (data): Attenuated detector counts

        unattenuated (data): Unattenuated detector counts

        source (detector|monitor): Measured tube [detector]

        mode (P|NP|mixed|auto): Dead-time mode [auto]

        **Returns**

        deadtime (deadtime): Deadtime constants

        2015-12-17 Paul Kienzle
    ``
    """
    return _parse_function(action)

timestamp = re.compile(r"^(?P<date>[0-9]{4}-[0-9]{2}-[0-9]{2})\s+(?P<author>.*?)\s*$")
def _parse_function(action):
    # Grab the docstring from the function
    # Note: inspect.getdoc() cleans the docstring, removing the indentation and
    # the leading and trailing carriage returns
    lines = inspect.getdoc(action).split('\n')

    # Default values for docstring portions
    description_lines = []
    input_lines = []
    output_lines = []
    version, author = "", ""

    # Split docstring into sections
    state = 0 # processing description
    for line in lines:
        match = timestamp.match(line)
        stripped = line.strip()
        if match is not None:
            state = 3
            version = match.group('date')
            author = match.group('author')
        elif stripped == "**Inputs**":
            state = 1
        elif line.strip() == "**Returns**":
            state = 2
        elif state == 0:
            description_lines.append(line)
        elif state == 1:
            input_lines.append(line)
        elif state == 2:
            output_lines.append(line)
        elif state == 3:
            raise ValueError("docstring continues after time stamp")
        else:
            raise RuntimeError("Unknown state %s"%state)

    # parse the sections
    description = "".join(description_lines).strip()
    inputs = _parse_parameters(input_lines)
    output_terminals = _parse_parameters(output_lines)

    # grab arguments and defaults from the function definition
    argspec = inspect.getargspec(action)
    args = argspec.args if not inspect.ismethod(action) else argspec.args[1:]
    defaults = (dict(zip(args[-len(argspec.defaults):], argspec.defaults))
                if argspec.defaults else {})
    if argspec.varargs is not None or argspec.keywords is not None:
        raise ValueError("function contains *args or **kwargs")

    # Check that all defined arguments are described
    defined = set(args)
    described = set(p['id'] for p in inputs)
    if defined-described:
        raise ValueError("Parameters defined but not described: "
                         + ",".join(sorted(defined-described)))
    if described-defined:
        raise ValueError("Parameters described but not defined: "
                         + ",".join(sorted(described-defined)))

    # Make sure there are no duplicates
    all_described = set(p['id'] for p in inputs+output_terminals)
    if len(all_described) != len(inputs)+len(output_terminals):
        raise ValueError("Parameter and return value names must be unique")

    # Split parameters into terminals (non-keyword) and fields (keyword)
    field_names = args[-len(defaults):] if defaults else []
    field_set = set(field_names)
    input_terminals = [p for p in inputs if p['id'] not in field_set]
    input_fields = [p for p in inputs if p['id'] in field_set]

    # Set the defaults for the fields from the keyword arguments
    for p in input_fields:
        if p['default'] is None:
            p['default'] = str(defaults[p['id']])

    # Set the terminal direction
    for p in input_terminals:
        p['use'] = 'in'
    for p in output_terminals:
        p['use'] = 'out'

    # Collect all the node info
    result = {
        'id': action.__name__,
        'name': _unsplit_name(action.__name__),
        'description': description,
        'terminals': input_terminals + output_terminals,
        'fields': input_fields,
        'version': version,
        'author': author,
        'module': action.__module__
        }

    return result

def _unsplit_name(name):
    """
    Convert "this_name" into "This Name".
    """
    return " ".join(s.capitalize() for s in name.split('_'))


# name (optional type): description [optional default]
parameter_re = re.compile("""\A
    \s*(?P<id>\w+)                       # name
    \s*([(]\s*(?P<type>[^)]*)\s*[)])?    # ( type )
    \s*:                                 # :
    \s*(?P<description>.*?)              # non-greedy description
    \s*([[]\s*(?P<default>.*?)\s*[]])?   # [ default ]
    \s*\Z""", re.VERBOSE)

def _parse_parameters(lines):
    """
    Interpret the doc strings for the parameters.

    Each parameter must use the form defined by parameter_re above:

        name (optional type): description [optional default]

    *lines* is the set of lines after ``**Inputs**`` and ``**Returns**``.
    Note that parameters are defined by consecutive non-blank lines separated
    by blank lines.  :func:`_group_parameters` is used to gather all of the
    relevant lines together, skipping the blank bits.
    """
    ret = []
    for group in _get_paragraphs(lines):
        s = " ".join(s.strip() for s in group)
        match = parameter_re.match(s)
        if match is None:
            raise ValueError("unable to parse parameter:\n  "+"  ".join(group))
        d = match.groupdict()
        d['required'] = False
        d['multiple'] = False
        if d['type'].endswith('?'):
            d['type'] = d['type'][:-1]
            d['required'] = True
        elif d['type'].endswith('*'):
            d['type'] = d['type'][:-1]
            d['multiple'] = True
        elif d['type'].endswith('+'):
            d['type'] = d['type'][:-1]
            d['required'] = True
            d['multiple'] = True
        d['label'] = _unsplit_name(d['id'])
        ret.append(d)
    return ret

def _get_paragraphs(lines):
    """
    Yield a list of paragraphs defined as lines separated by blank lines.

    Each paragraph is returned as a list of lines.
    """
    group = []
    for line in lines:
        if line.strip() == "":
            if group:
                yield group
            group = []
        else:
            group.append(line)
    if group:
        yield group