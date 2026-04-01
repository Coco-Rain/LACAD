import cadquery as cq

result = (
cq.Workplane("XY")
.box(10, 10, 10)
.faces(">X")
.workplane()
.circle(2)
.extrude(5)
)
cq.exporters.export(result, 'GT.stl')