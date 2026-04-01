import cadquery as cq

result = (
cq.Workplane("XY")
.box(8, 6, 4)
.faces(">Z")
.workplane()
.sketch()
.circle(1.5)
.finalize()
.extrude(3)
)
cq.exporters.export(result, 'GT.stl')