#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
# 
#  License for the specific language governing rights and limitations under
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included
#  in all copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
#

#    File   : Unu3op.tcl
#    Author : Martin Cole
#    Date   : Mon Sep  8 09:46:23 2003

catch {rename Teem_UnuAtoM_Unu3op ""}

itcl_class Teem_UnuAtoM_Unu3op {
    inherit Module
    constructor {config} {
        set name Unu3op
        set_defaults
    }
    method set_defaults {} {
        global $this-operator
        set $this-operator "+"

	global $this-float1
	set $this-float1 "0.0"

	global $this-float2
	set $this-float2 "0.0"

	global $this-float3
	set $this-float3 "0.0"
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }

        toplevel $w

        frame $w.f
	pack $w.f -padx 2 -pady 2 -side top -expand yes
	
	frame $w.f.options
	pack $w.f.options -side top -expand yes

        iwidgets::entryfield $w.f.options.operator -labeltext "Operator:" -textvariable $this-operator
        pack $w.f.options.operator -side top -expand yes -fill x
	
	iwidgets::entryfield $w.f.options.float1 \
	    -labeltext "Float Input 1:" -textvariable $this-float1
        pack $w.f.options.float1 -side top -expand yes -fill x

	iwidgets::entryfield $w.f.options.float2 \
	    -labeltext "Float Input 2:" -textvariable $this-float2
        pack $w.f.options.float2 -side top -expand yes -fill x

	iwidgets::entryfield $w.f.options.float3 \
	    -labeltext "Float Input 3:" -textvariable $this-float3
        pack $w.f.options.float3 -side top -expand yes -fill x

	makeSciButtonPanel $w.f $w $this
	moveToCursor $w

	pack $w.f -expand 1 -fill x
    }
}
