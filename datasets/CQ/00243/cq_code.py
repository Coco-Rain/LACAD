import cadquery as cq

result = (
cq.Workplane("front")
.sphere(8)
.intersect(
cq.Workplane("XY")
.box(12, 12, 12)
.val(),
clean=True
)
)
cq.exporters.export(result, 'GT.stl')