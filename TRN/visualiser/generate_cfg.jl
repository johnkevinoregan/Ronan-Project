

# A simple CFG generator for specific functions in TRM_core.jl
# It uses Meta.parse to avoid needing project dependencies installed.

# --- Graphviz / DOT Generation Helpers ---
abstract type Node end

mutable struct Block
    id::Int
    label::String
    stmts::Vector{Any} # Expressions
    next::Vector{Block}
end

Block(id::Int, label::String) = Block(id, label, [], [])

struct CFG
    name::String
    blocks::Vector{Block}
    entry::Block
    exit::Block
    edges::Vector{Pair{Int, Int}} # (from_id, to_id)
end

function CFG(name::String)
    entry = Block(1, "Entry")
    exit = Block(2, "Exit")
    CFG(name, [entry, exit], entry, exit, [])
end

function add_block!(cfg::CFG, label::String)
    id = length(cfg.blocks) + 1
    b = Block(id, label)
    push!(cfg.blocks, b)
    return b
end

function add_edge!(cfg::CFG, from::Block, to::Block; label="")
    # We store basic edges.
    push!(cfg.edges, from.id => to.id)
    # Also update block next pointers for our traversal if needed
    push!(from.next, to)
end

# --- AST Walker ---

function extract_functions(expr::Expr)
    funcs = Dict{String, Expr}()
    
    # Helper to walk
    function walk(ex)
        if ex isa Expr
            if ex.head == :function || (ex.head == :(=) && ex.args[1] isa Expr && ex.args[1].head == :call)
                # Found a function
                sig = ex.args[1]
                val_name = if sig.head == :call
                    sig.args[1]
                elseif sig.head == :where
                     sig.args[1].args[1] # handle param types like foo(x::T) where T
                else
                    nothing
                end

                if val_name isa Symbol
                    funcs[string(val_name)] = ex
                elseif val_name isa Expr && val_name.head == :. # module.func?
                     # simplify for now
                end
            elseif ex.head == :module
                # Recurse into module
                for arg in ex.args
                    walk(arg)
                end
            else
                for arg in ex.args
                    walk(arg)
                end
            end
        end
    end
    
    walk(expr)
    return funcs
end

function build_cfg(func_name::String, func_expr::Expr)
    cfg = CFG(func_name)
    current_block = cfg.entry
    
    # Body is usually the second argument
    body = func_expr.args[2]
    
    function process_block(exprs, parent_block, exit_block)
        current = parent_block
        
        for ex in exprs
            if ex isa LineNumberNode
                continue
            end
            
            if ex isa Expr
                if ex.head == :for
                    # Loop structure: for iter = range; body; end
                    loop_head = add_block!(cfg, "Loop Start: $(ex.args[1])")
                    add_edge!(cfg, current, loop_head)
                    
                    loop_body_start = add_block!(cfg, "Loop Body")
                    add_edge!(cfg, loop_head, loop_body_start; label="enter")
                    
                    # Recurse for body
                    loop_body_end = process_block(ex.args[2].args, loop_body_start, loop_head)
                    
                    # Edge back to head
                    add_edge!(cfg, loop_body_end, loop_head; label="repeat")
                    
                    # After loop
                    after_loop = add_block!(cfg, "After Loop")
                    add_edge!(cfg, loop_head, after_loop; label="exit")
                    
                    current = after_loop
                    
                elseif ex.head == :block
                     current = process_block(ex.args, current, exit_block)
                elseif ex.head == :call
                    # Function call
                     stmt_node = add_block!(cfg, "Call: $(ex.args[1])")
                     add_edge!(cfg, current, stmt_node)
                     current = stmt_node
                 elseif ex.head == :(=)
                     # Assignment
                     lhs = ex.args[1]
                     rhs = ex.args[2]
                     # Check if RHS is a call or block
                     label = "Assign: $lhs"
                     if rhs isa Expr && rhs.head == :call
                         label *= " = $(rhs.args[1])(...)"
                     end
                     stmt_node = add_block!(cfg, label)
                     add_edge!(cfg, current, stmt_node)
                     current = stmt_node
                elseif ex.head == :return
                    ret_node = add_block!(cfg, "Return")
                    add_edge!(cfg, current, ret_node)
                    add_edge!(cfg, ret_node, cfg.exit)
                    # Control flow ends here for this path
                    return cfg.exit 
                else
                    # Generic statement
                    label = "Stmt: $(ex.head)"
                    stmt_node = add_block!(cfg, label)
                    add_edge!(cfg, current, stmt_node)
                    current = stmt_node
                end
            else
                # Literal or symbol
            end
        end
        return current
    end
    
    if body isa Expr && body.head == :block
        final_block = process_block(body.args, current_block, cfg.exit)
        if final_block != cfg.exit
             add_edge!(cfg, final_block, cfg.exit)
        end
    else
        # Single line function?
        final_block = process_block([body], current_block, cfg.exit)
         if final_block != cfg.exit
             add_edge!(cfg, final_block, cfg.exit)
        end
    end
    
    return cfg
end


function to_dot(cfgs::Vector{CFG})
    buf = IOBuffer()
    println(buf, "digraph TRM_Flow {")
    println(buf, "  node [shape=box, style=filled, fillcolor=\"#f0f0f0\", fontname=\"Helvetica\"];")
    println(buf, "  edge [fontname=\"Helvetica\"];")
    println(buf, "  compound=true;")
    
    for (i, cfg) in enumerate(cfgs)
        println(buf, "  subgraph cluster_$(i) {")
        println(buf, "    label = \"$(cfg.name)\";")
        println(buf, "    style = rounded;")
        println(buf, "    color = lightgrey;")
        
        for b in cfg.blocks
            label = b.label
            # Escape quotes
            label = replace(label, "\"" => "\\\"")
            shape = if b.label == "Entry" || b.label == "Exit" "oval" else "box" end
            color = if b.label == "Entry" "lightgreen" elseif b.label == "Exit" "lightpink" else "#f9f9f9" end
            println(buf, "    node$(i)_$(b.id) [label=\"$(label)\", shape=$shape, fillcolor=\"$color\"];")
        end
        
        for (u, v) in cfg.edges
            println(buf, "    node$(i)_$(u) -> node$(i)_$(v);")
        end
        
        println(buf, "  }")
    end
    
    println(buf, "}")
    return String(take!(buf))
end

# --- Main Execution ---

function main()
    filename = "TRM_core.jl"
    if !isfile(filename)
        println("Error: File $filename not found in $(pwd())")
        exit(1)
    end
    
    println("Parsing $filename...")
    content = read(filename, String)
    ast = Meta.parse(content)
    
    println("Extracting functions...")
    funcs = extract_functions(ast)
    println("Found functions: ", keys(funcs))
    
    targets = ["forward_inner", "deep_recursion"]
    cfgs = CFG[]
    
    for t in targets
        if haskey(funcs, t)
            println("Building CFG for $t...")
            push!(cfgs, build_cfg(t, funcs[t]))
        else
            println("Warning: Function $t not found.")
        end
    end
    
    dot_content = to_dot(cfgs)
    outfile = "trm_cfg.dot"
    write(outfile, dot_content)
    println("DOT file written to $outfile")
end

main()
