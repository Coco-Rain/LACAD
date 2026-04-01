import cadquery as cq

result = (
cq.Workplane("XY")
.box(10, 10, 2)
.faces(">Z")
.workplane()
.circle(3)
.extrude(2)
.solids(">Z")
.faces(">Z")
.chamfer(0.5)
)
cq.exporters.export(result, 'GT.stl')