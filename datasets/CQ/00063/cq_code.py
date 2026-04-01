import cadquery as cq

result = (
cq.Workplane("XY")
.box(6, 8, 2)
.faces(">Z")
.workplane()
.center(0, -3)
.ellipse(2.5, 1.5)
.extrude(1.5)
)
cq.exporters.export(result, 'GT.stl')