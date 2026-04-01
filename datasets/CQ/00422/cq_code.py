import cadquery as cq

result = (
cq.Workplane("XY")
.box(8, 6, 2)
.faces(">Z")
.workplane()
.center(3, 0)
.ellipse(1.5, 0.5)
.cutThruAll()
)
cq.exporters.export(result, 'GT.stl')