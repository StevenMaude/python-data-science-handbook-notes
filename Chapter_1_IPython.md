# Chapter 1: IPython

## General help

`thing?` shows docstring of `thing`.

`thing??` shows source of `thing` (unless in some compiled language,
e.g. C extensions).

## Tab completion

### Of object attributes and methods

`obj.<TAB>` or `obj._<TAB>` for internal attributes/methods.

## Wildcard matching

`*Warning?` lists objects in the namespace that end with Warning.

`*` matches any string, including ''.

`str.*find*?`

## Shortcuts

### Navigation

Ctrl-a: Move cursor to beginning of line.  
Ctrl-e: Move cursor to end of line.  
Ctrl-b (or left arrow): Move cursor back one character.  
Ctrl-f (or right arrow): Move cursor forward one character.  

### Text entry

Ctrl-d: Delete next character in line.  
Ctrl-k: Cut text from cursor to end of line.  
Ctrl-u: Cut text from beginning of line to cursor.  
Ctrl-y: Yank (paste) text that was cut.  
Ctrl-t: Transpose previous two characters.  

### Command history

Ctrl-p (or up arrow): Access previous command in history.  
Ctrl-n (or down arrow): Access next command in history.  
Ctrl-r: Reverse search through history.  

### Miscellaneous

Ctrl-l: Clear terminal screen.   
Ctrl-c: Interrupt Python command.  
Ctrl-d: Exit IPython.   

## Magic functions

Enhancements over standard Python shell.

Prefixed by `%`.

Line magics: single `%`, operate on a line of input. Function gets this
line as an argument.

Cell magics: double `%%`, operate on multiple input lines. Function gets
the first line as an argument, and the lines below as a separate
argument.

### Help for magic functions

See `%magic` and `%lsmagic`.

### Pasting code blocks

`%paste`: pasting preformatted code block from elsewhere, e.g. with
          interpreter markers.

`%cpaste`: similar, can paste one or more chunks
           of code to run in batch.

### Running external code

`%run`: Runs Python file as a program. Anything defined in there, is
then available in IPython subsequently (unless you run the code with the
profiler via -p).

### Timing code execution

`%timeit` times how long a line of code takes to execute.

`%%timeit` is the cell magic version, can specify multiple lines of code
to execute.

## Input and output history

### In and Out

`In` is a list of commands.

`Out` is a dictionary that maps input numbers to any outputs.

Can use these to reuse commands or outputs, useful if an output takes a
long time to compute.

**NB: not everything gives an Out value (e.g. a function call that
returns None).**

### Underscore shortcuts

`_` access last output (also works in standard Python shell).  
`__`access penultimate output.  
`___` access propenultimate output.  

`_X` is a shortcut for `Out[X]`.

### Suppress output

Use a semicolon at end of command. Either, e.g. useful for plotting
commands, or to allow the output to be deallocated.

Doesn't display and doesn't get added to `Out`.

### Related magic commands

`%history -n 1-4`: print first four inputs.

`%rerun`: rerun some portion of command history.

`%save`: save some portion of command history to file.

## IPython and the system command shell

### Running commands

Anything following `!` on a line executed by the system command shell,
not IPython.

### Passing data from system shell to IPython

Can use this to interact with IPython, e.g. `contents = !ls`. Such a
"list" isn't a Python list, but a special IPython shell type; these have
`grep`, `fields` methods, and `s`, `n` and `p` properties to search,
filter and display results.

### Passing data from IPython to system shell

Using `{varname}` substitutes that Python variable's value into the
command.

`message = "hello from Python"`

and then you can:

`!echo {message}`

Can't use `!cd` to navigate, because commands are in a subshell. Need to
use `%cd` or can even just do `cd` which is an `automagic` function. Can
toggle this behaviour with `%automagic`.

Other shell-like magic functions: `%cat`, `%cp`, `%env`, `%ls`, `%man`,
`%mkdir`, `%more`, `%mv`, `%pwd`, `%rm`, `%rmdir`. These can all be used
without `%` if `automagic` is enabled.

## Errors and debugging

### Controlling exceptions: `%xmode`

x is for *exception*. Changes reporting:

`%xmode Plain` (less information)  
`%xmode Context` (default)  
`%xmode Verbose` (more information, displays arguments to functions)  

### Debugging: `ipdb`

`ipdb` is the IPython version of `pdb`, the Python debugger.

Using `%debug` immediately following an exception opens a debugging
prompt at the point of the exception.

In the `ipdb>` prompt, can do `up` or `down` to move through the stack
and explore variables there.

`%pdb on` enables the debugger by default when an exception is raised.

#### Debugger commands

`list`          Show current location in file.  
`h(elp)`        Show list of commands, or get help on current command.  
`q(uit)`        Quit debugger and program.  
`c(ontinue)`    Quit debugger, continue program.  
`n(ext)`        Continue until next line in current function is reached
                or it returns; called functions are executed without
                stopping.  
`<ENTER>`       Repeat previous command.  
`p(rint)`       Print variables.  
`s(tep)`        Continue, but stop as soon as possible (whether in a
                called function or in current function).  
`r(eturn)`      Continue until function returns.  

### Stepping through code

`%run -d` your script, and then use `next` to move through lines of
code.

## Profiling and timing code

### Timing lines of code

`%timeit` (line) and `%%timeit` (cell).

By default, these repeat the code.

Repeating sometimes is misleading, e.g. sorting a sorted list.

Instead, can use `%time` to run once.

Also: `%timeit` prevents system calls from interfering with timing.

### Profiling a statement: `%prun`

e.g. `%prun <function>`

Tells you how long programs spends in particular function calls.

### Line-by-line profiling: `%lprun`

This requires the `line_profiler` extension:

`pip install line_profiler`

and then load this extension:

`%load_ext line_profiler`

then:

`%lprun -f <function>`

to profile the code. Shows you specific lines and how long the program
spends completing them.

### Memory usage: `%memit` and `%mprun`

This requires the `memory_profiler` extension:

`pip install memory_profiler`

and then load this extension:

`%load_ext memory_profiler`

then:

`%memit <function>` (like `%timeit`)

or:

`%mprun <function>` (like `%prun`, line-by-line)

to profile the code.

`%mprun` requires a function in a file, but can use `%%file <filename>`
to create a new file with content.
