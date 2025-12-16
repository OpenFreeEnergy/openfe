# Contributing CLI Subcommands

Adding a new subcommand to the `openfe` CLI is pretty straightforward, but
there are some best practices that will make your contribution easier to
maintain.

## How the CLI finds subcommands

Subcommands are registered with the CLI based on the existence of an instance
of `CommandPlugin` in modules located in particular directories or namespaces.
This means that after you create the command function, you need to wrap it in a
`CommandPlugin`, which must be assigned to a variable name. The variable name
itself is unimportant (I usually use `PLUGIN`). It's perfectly fine to include
more than one plugin in the same file, but the each must have a different
variable name.

The allowed locations for command plugins may change, but currently includes:

* modules located in the namespace `openfecli.commands`

When contributing to the core CLI, all you need to do is add your subcommand
module to the `openfecli/commands/` directory, and the CLI should register it
automatically.

## Best practices

### The CLI should be a thin wrapper around the library

The intent of the CLI is to provide convenient ways of accomplishing things
that can also be accomplished with the core library. This means that CLI
commands should be thin wrappers that either just call a method from the core
library, or run a very simple workflow based on methods from the core library.

If you find that your CLI command starts to have some more complex logic, this
probably means that some of that logic would be beneficial to users of the
library as well. Consider moving that code into the core library.

This also implies that we can split any CLI subcommand into two stages:

1. Convert from user input to objects that have meaning to the library.
2. Run some code as if we were users of the library, with no reference to the
   fact that the inputs came from the command line.

### Divide the subcommand module into three components

The recommended way of structuring a subcommand module is to split it into
three parts (where `command` is replaced by the name of your subcommand):

1. `command`: The command method, which is decorated by `@click.command`. The
   purpose of this method is to convert user input to objects that can be used
   by the core library. Then it calls the `command_main` method.
2. `command_main`: The workflow method, which is written using code from the
   underlying library, with no reference to the fact that this is part of the
   CLI. This typically contains a very simple workflow script. Although the
   output from this process is usually saved to some output file as part of the
   script in `command_main`, the best practice is to also return the result of
   this method. The `command` method will ignore this return value, but
   returning it makes it so that the `command_main` method can be reused in
   other CLI commands to create more complex workflows.
3. `PLUGIN`: a `CommandPlugin` instance, which wraps the `command` object with
   metadata about the subcommand, such as which help section to display it in,
   and which versions of the library and CLI the plugin is compatible with.

As an example, here's a rough skeleton for a subcommand called `my_command`
(imports excluded)

```python
@click.command("my_command", short_help="This is my command")
...  # add decorators for arguments/options
def my_command(...):  # input params are based on arguments options
    """Docstring here is the help given by ``openfe my-command --help``"""
    ...  # do whatever you need to convert the user input to library objects
    my_command_main(...)  # takes library objects

def my_command_main(...):  # takes library objects
    ...  # run some simple library code
    return result

PLUGIN = CommandPlugin(
    command=my_command,
    section="My Section",
    requires_lib=(1, 0),
    requires_cli=(1, 0)
)
```

### Use reusable subcommand arguments/options

In `click`, command-line arguments and options are declared by attaching
decorators for each option to a method. The method must then take parameters
based on the option name as specified by the decorator.

Because of this, it is straightforward to create an object associated with a
given input option/argument, which contains details such as the help string and
even a method to get a library object from the user input string.

The best practice is to create this object outside a given subcommand, and then
reuse it between different subcommands. This ensures that the user sees
consistency in the interface and behavior between different CLI subcommands.

Details on how we'll do this in OpenFE are still being developed.

### Delay slow imports

Usually in Python, we put all imports at the top of a file. That is the best
practice for libraries and scripts, because it makes it easy for a developer to
find dependencies, and helps prevent developers from repeating import
statements.

However, when dealing with a CLI script like this, it's important to remember
that some user interactions, such as subcommand autocomplete or enquiring about the
CLI with `--version` or `--help`, will also run any top-level imports. If
imports are slow, then these user-facing interactions will be slow.

Because of this, the best practice when writing CLI subcommands is to move slow
imports inside the method that needs them.

### Testing your subcommand

Dividing the subcommand as recommended above facilitates testing. When testing
the `command` method itself, mock out the `command_main`, and use tools within
`click` to mock the user command inputs. The purpose of testing the `command`
method is to ensure that you correctly convert from user input to library object.

The purpose of testing the `command_main` method is to ensure that integration
with the library works. If this is truly a thin wrapper (and with the
assumption that the core library is thoroughly tested), then a smoke test may
be sufficient for `command_main`.
