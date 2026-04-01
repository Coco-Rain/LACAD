import cadquery as cq

result = (
cq.Workplane("XY")
.box(8, 6, 10)
.faces(">Y")
.workplane()
.circle(3)
.extrude(-2)
.faces(">Y")
.vals()
)
cq.exporters.export(result, 'GT.stl')