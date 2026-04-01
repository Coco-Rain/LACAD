import cadquery as cq

result = (
cq.Workplane("XY")
.box(30, 25, 15)
.union(
cq.Workplane("XZ")
.workplane(offset=12.5)
.ellipse(10, 6)
.extrude(5)
)
.faces(">Y")
.shell(3)
)
cq.exporters.export(result, 'GT.stl')