import cadquery as cq

result = (
cq.Workplane("XY")
.box(8, 8, 2)
.faces(">Z")
.workplane()
.sketch()
.circle(2)
.finalize()
.extrude(3)
)
cq.exporters.export(result, 'GT.stl')