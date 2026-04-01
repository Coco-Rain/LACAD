import cadquery as cq

result = (
cq.Workplane("XY")
.box(8, 6, 4)
.faces(">Z").workplane()
.center(1, -1)
.circle(1.5)
.last()
.extrude(2, combine=False)
)
cq.exporters.export(result, 'GT.stl')