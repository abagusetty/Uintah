#
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


catch {rename VolumeVisualizer ""}

itcl_class Volume_Visualization_VolumeVisualizer {
    inherit Module
    constructor {config} {
	set name VolumeVisualizer
	set_defaults
    }
    method set_defaults {} {
	global $this-num_slices
	set $this-num_slices 64
	global $this-alpha_scale
	set $this-alpha_scale 0
	global $this-render_style
	set $this-render_style 0
	global $this-interp_mode 
	set $this-interp_mode 1
	global $this-contrast
	set $this-contrast 0.5
	global $this-contrastfp
	set $this-contrastfp 0.5
        global $this-shading
        set $this-shading 0
        global $this-ambient
        set $this-ambient 0.5
        global $this-diffuse
        set $this-diffuse 0.5
        global $this-specular
        set $this-specular 0.0
        global $this-shine
        set $this-shine 30.0
        global $this-light
        set $this-light 0
    }
    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    return
	}
	toplevel $w
	wm minsize $w 250 300
	frame $w.f -relief groove -borderwidth 2 
	pack $w.f -padx 2 -pady 2 -fill x
	set n "$this-c needexecute"
        set s "$this state"

	global $this-render_style
	label $w.f.l -text "Rendering Style"
	radiobutton $w.f.modeo -text "Over Operator" -relief flat \
		-variable $this-render_style -value 0 \
		-anchor w -command $n
	radiobutton $w.f.modem -text "MIP" -relief flat \
		-variable $this-render_style -value 1 \
		-anchor w -command $n
# 	radiobutton $w.f.modea -text "Attenuate" -relief flat \
# 		-variable $this-render_style -value 2 \
# 		-anchor w -command $n
	pack $w.f.l $w.f.modeo $w.f.modem -side top -fill x -padx 4 -pady 4

	frame $w.f4 -relief groove -borderwidth 2
	pack $w.f4 -padx 2 -pady 2 -fill x
	checkbutton $w.f4.shading -text "Shading" -relief flat \
            -variable $this-shading -onvalue 1 -offvalue 0 \
            -anchor w -command "$s; $n"
	global $this-ambient
	scale $w.f4.ambient -variable $this-ambient \
            -from 0.0 -to 1.0 -label "Ambient" \
            -showvalue true -resolution 0.001 \
            -orient horizontal
	global $this-diffuse
	scale $w.f4.diffuse -variable $this-diffuse \
		-from 0.0 -to 1.0 -label "Diffuse" \
		-showvalue true -resolution 0.001 \
		-orient horizontal
	global $this-specular
	scale $w.f4.specular -variable $this-specular \
		-from 0.0 -to 1.0 -label "Specular" \
		-showvalue true -resolution 0.001 \
		-orient horizontal
	global $this-shine
	scale $w.f4.shine -variable $this-shine \
		-from 1.0 -to 128.0 -label "Shine" \
		-showvalue true -resolution 1.0 \
		-orient horizontal
        pack $w.f4.shading $w.f4.ambient $w.f4.diffuse \
            $w.f4.specular $w.f4.shine \
            -side top -fill x -padx 4

	frame $w.f3 -relief groove -borderwidth 2
	pack $w.f3 -padx 2 -pady 2 -fill x
	label $w.f3.l -text "Interpolation Mode"
	radiobutton $w.f3.interp -text "Trilinear" -relief flat \
		-variable $this-interp_mode -value 1 \
		-anchor w -command $n
	radiobutton $w.f3.near -text "Nearest" -relief flat \
		-variable $this-interp_mode -value 0 \
		-anchor w -command $n
	pack $w.f3.l $w.f3.interp $w.f3.near -side top -fill x -padx 4


# 	frame $w.f5 -relief groove -borderwidth 2
# 	pack $w.f5 -padx 2 -pady 2 -fill x
# 	label $w.f5.light -text "Light" -justify left
# 	radiobutton $w.f5.light0 -text "Light 0" -relief flat \
# 		-variable $this-light -value 0 \
# 		-anchor w -command $n
# 	radiobutton $w.f5.light1 -text "Light 1" -relief flat \
# 		-variable $this-light -value 1 \
# 		-anchor w -command $n
#         pack $w.f5.light $w.f5.light0 $w.f5.light1 \
#             -side top -fill x -padx 4

	global $this-num_slices
	scale $w.nslice -variable $this-num_slices \
		-from 1 -to 1024 -label "Number of Slices" \
		-showvalue true \
		-orient horizontal \

	global $this-alpha_scale
	
	scale $w.stransp -variable $this-alpha_scale \
		-from -1.0 -to 1.0 -label "Slice Transparency" \
		-showvalue true -resolution 0.001 \
		-orient horizontal 

	pack $w.stransp $w.nslice  -side top -fill x -padx 4 -pady 2

        bind $w.f4.ambient <ButtonRelease> $n
        bind $w.f4.diffuse <ButtonRelease> $n
        bind $w.f4.specular <ButtonRelease> $n
        bind $w.f4.shine <ButtonRelease> $n

	bind $w.nslice <ButtonRelease> $n
	bind $w.stransp <ButtonRelease> $n
	
	makeSciButtonPanel $w $w $this
        $this state
	moveToCursor $w
    }

    method state {} {
	set w .ui[modname]
	if {[set $this-shading] == 1} {
            $this activate $w.f4.ambient
            $this activate $w.f4.diffuse
            $this activate $w.f4.specular
            $this activate $w.f4.shine
#             $this activate $w.f5.light
#             $this activate $w.f5.light0
#             $this activate $w.f5.light1
	} else {
            $this deactivate $w.f4.ambient
            $this deactivate $w.f4.diffuse
            $this deactivate $w.f4.specular
            $this deactivate $w.f4.shine
#             $this deactivate $w.f5.light
#             $this deactivate $w.f5.light0
#             $this deactivate $w.f5.light1
	}
    }
    method activate { w } {
	$w configure -state normal -foreground black
    }
    method deactivate { w } {
	$w configure -state disabled -foreground darkgrey
    }


}
