#  The contents of this file are subject to the University of Utah Public
#  License (the "License"); you may not use this file except in compliance
#  with the License.
#  
#  Software distributed under the License is distributed on an "AS IS"
#  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
#  License for the specific language governing rights and limitations under
#  the License.
#  
#  The Original Source Code is SCIRun, released March 12, 2001.
#  
#  The Original Source Code was developed by the University of Utah.
#  Portions created by UNIVERSITY are Copyright (C) 2001, 1994
#  University of Utah. All Rights Reserved.
#  
#    File   : axis_info_sel_box.tcl
#    Author : Martin Cole
#    Date   : Fri Feb 21 10:16:34 2003

global teem_num_axes
set teem_num_axes 0


proc get_selection {w} {
    set f [$w childsite]
    $f.rb get
}

proc make_axis_info_sel_box {w command} {
    
    iwidgets::scrolledframe $w -relief groove -width 250 -height 200 \
		-labelpos nw -labeltext "Axis Info and Selection"

    pack $w -expand yes -fill both -side top
    set f [$w childsite]

    iwidgets::radiobox $f.rb -relief flat -command $command
    pack $f.rb -side top -fill x -fill y -expand yes
}

proc add_axis {w tag info} {
    global teem_num_axes
    set f [$w childsite]
    $f.rb add $tag -text $info
    incr teem_num_axes
    puts $teem_num_axes
}

proc delete_all_axes {w} {
    global teem_num_axes
    set f [$w childsite]

    while {$teem_num_axes > 0} {
	set teem_num_axes [expr $teem_num_axes - 1]
	$f.rb delete $teem_num_axes
    }
}