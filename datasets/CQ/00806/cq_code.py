import cadquery as cq

result = (
cq.Workplane("XY")
.box(50, 40, 5)
.faces(">Z")
.workplane()
.sketch()
.rarray(20, 15, 3, 2)
.circle(2)
.finalize()
.cutThruAll()
)
cq.exporters.export(result, 'GT.stl')