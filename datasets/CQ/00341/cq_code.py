import cadquery as cq

result = (
cq.Workplane("XY")
.box(30, 20, 4)
.faces(">Z")
.workplane()
.sketch()
.trapezoid(12, 6, 60, 60)
.finalize()
.extrude(2)
)
cq.exporters.export(result, 'GT.stl')