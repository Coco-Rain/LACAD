import cadquery as cq

result = (
cq.Workplane("XY")
.box(30, 25, 15)
.faces(">Z")
.workplane()
.ellipse(12, 8)
.extrude(5)
.faces(">Z")
.workplane()
.center(-10, 5)
.cskHole(4.5, 8.0, 6, 82)
)
cq.exporters.export(result, 'GT.stl')