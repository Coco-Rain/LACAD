import cadquery as cq

result = (
cq.Workplane("XY")
.box(10, 10, 1)
.faces(">Z")
.workplane()
.center(2, 2)
.circle(1)
.extrude(1)
.translate((3, 3, 0))
)
cq.exporters.export(result, 'GT.stl')