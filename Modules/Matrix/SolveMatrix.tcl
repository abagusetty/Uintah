
itcl_class SolveMatrix {
    inherit Module
    constructor {config} {
	set name SolveMatrix
	set_defaults
    }
    method set_defaults {} {
	global $this-target_error $this-method $this-orig_error
	global $this-current_error $this-flops $this-floprate $this-iteration
	global $this-memrefs $this-memrate $this-maxiter
	global $this-use_previous_soln
	set $this-target_error 1.e-4
	set $this-method conjugate_gradient
	set $this-orig_error 9.99999e99
	set $this-current_error 9.99999e99
	set $this-flops 0
	set $this-floprate 0
	set $this-memrefs 0
	set $this-memrate 0
	set $this-iteration 0
	set $this-maxiter 0
	set $this-use_previous_soln 1
    }
    method ui {} {
	set w .ui$this
	if {[winfo exists $w]} {
		raise $w
		return;
	}

	toplevel $w
	wm minsize $w 300 20
	set n "$this-c needexecute "

	button $w.execute -text "Execute" -command $n
	pack $w.execute -side top -fill x -pady 2 -padx 2

	make_labeled_radio $w.method "Solution Method" "" \
		top $this-method \
		{{"Jacobi" jacobi} \
		{"Conjugate Gradient" conjugate_gradient}}

	pack $w.method -side top -fill x -pady 2

	expscale $w.target_error -orient horizontal -label "Target error:" \
		-variable $this-target_error -command ""
	pack $w.target_error -side top -fill x -pady 2

	scale $w.maxiter -orient horizontal -label "Maximum Iterations:" \
		-variable $this-maxiter -from 0 -to 200
	pack $w.maxiter -side top -fill x -pady 2

	checkbutton $w.use_prev -variable $this-use_previous_soln \
		-text "Use previous solution as initial guess"
	pack $w.use_prev -side top -fill x -pady 2


	frame $w.converg -borderwidth 2 -relief ridge
	pack $w.converg -side top -padx 2 -pady 2 -fill x

	frame $w.converg.iter
	pack $w.converg.iter -side top -fill x
	label $w.converg.iter.lab -text "Iteration: "
	pack $w.converg.iter.lab -side left
	label $w.converg.iter.val -textvariable $this-iteration
	pack $w.converg.iter.val -side right

	frame $w.converg.first
	pack $w.converg.first -side top -fill x
	label $w.converg.first.lab -text "Original Error: "
	pack $w.converg.first.lab -side left
	label $w.converg.first.val -textvariable $this-orig_error
	pack $w.converg.first.val -side right

	frame $w.converg.current
	pack $w.converg.current -side top -fill x
	label $w.converg.current.lab -text "Current Error: "
	pack $w.converg.current.lab -side left
	label $w.converg.current.val -textvariable $this-current_error
	pack $w.converg.current.val -side right

	frame $w.converg.flopcount
	pack $w.converg.flopcount -side top -fill x
	label $w.converg.flopcount.lab -text "Flop Count: "
	pack $w.converg.flopcount.lab -side left
	label $w.converg.flopcount.val -textvariable $this-flops
	pack $w.converg.flopcount.val -side right

	frame $w.converg.floprate
	pack $w.converg.floprate -side top -fill x
	label $w.converg.floprate.lab -text "MFlops: "
	pack $w.converg.floprate.lab -side left
	label $w.converg.floprate.val -textvariable $this-floprate
	pack $w.converg.floprate.val -side right

	frame $w.converg.memcount
	pack $w.converg.memcount -side top -fill x
	label $w.converg.memcount.lab -text "Memory bytes accessed: "
	pack $w.converg.memcount.lab -side left
	label $w.converg.memcount.val -textvariable $this-memrefs
	pack $w.converg.memcount.val -side right

	frame $w.converg.memrate
	pack $w.converg.memrate -side top -fill x
	label $w.converg.memrate.lab -text "Memory bandwidth (MB/sec):"
	pack $w.converg.memrate.lab -side left
	label $w.converg.memrate.val -textvariable $this-memrate
	pack $w.converg.memrate.val -side right

	global $this-target_error
	set err [set $this-target_error]

	blt_graph $w.graph -title "Convergence" -height 250 \
		-plotbackground gray70
	$w.graph yaxis configure -logscale true -title "error (RMS)"
	$w.graph xaxis configure -title "Iteration" \
		-loose true

	bind $w.graph <ButtonPress-1> "$this select_error %x %y"
	bind $w.graph <Button1-Motion> "$this move_error %x %y"
	bind $w.graph <ButtonRelease-1> "$this deselect_error %x %y"

	set iter 1
	$w.graph element create "Current Target" -linewidth 1
	$w.graph element configure "Current Target" -data "0 $err" \
		-symbol diamond

	pack $w.graph -fill x
    }
    protected error_selected false
    protected tmp_error
    method select_error {wx wy} {
	global $this-target_error $this-iteration
	set w .ui$this
	set err [set $this-target_error]
	set iter [set $this-iteration]
	set errpos [$w.graph transform $iter $err]
	set erry [lindex $errpos 1]
	set errx [lindex $errpos 0]
	if {abs($wy-$erry)+abs($wx-$errx) < 5} {
	    $w.graph element configure "Current Target" -foreground yellow
	    set error_selected true
	}
    }
    method move_error {wx wy} {
	set w .ui$this
	set newerror [lindex [$w.graph invtransform $wx $wy] 1]
	
	$w.graph element configure "Current Target" -ydata $newerror
    }
    method deselect_error {wx wy} {
	set w .ui$this

	$w.graph element configure "Current Target" -foreground blue
	set error_selected false

	set newerror [lindex [$w.graph invtransform $wx $wy] 1]
	global $this-target_error
	set $this-target_error $newerror
    }
    protected min_error
    method reset_graph {} {
	set w .ui$this
	if {![winfo exists $w]} {
	    return
	}
	catch "$w.graph element delete {Target Error}"
	catch "$w.graph element delete {Current Error}"
	$w.graph element create "Target Error" -linewidth 0 -foreground blue
	$w.graph element create "Current Error" -linewidth 0 -foreground red 
	global $this-target_error
	set err [set $this-target_error]
	set iter 1
	$w.graph element configure "Target Error" -data "0 $err $iter $err"
	set min_error $err
    }

    method append_graph {iter values errvalues} {
	set w .ui$this
	if {![winfo exists $w]} {
	    return
	}
	if {$values != ""} {
	    $w.graph element append "Current Error" "$values"
	}
	if {$errvalues != ""} {
	    $w.graph element append "Target Error" "$errvalues"
	}
	global $this-target_error
	set err [set $this-target_error]
	if {$err < $min_error} {
	    set min_error $err
	}
	$w.graph yaxis configure -min [expr $min_error/10]
	$w.graph element configure "Current Target" -xdata $iter
    }

    method finish_graph {} {
	set w .ui$this
	$w.graph element configure "Current Error" -foreground green
    }
}
