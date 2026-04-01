import cadquery as cq

result = cq.Workplane("XY").box(3, 7, 7)
bbox = result.faces(">Y").val().BoundingBox()
result = (
result.faces(">Y")
.workplane()
.moveTo(bbox.center.x, bbox.center.y)
.box(1,1,1)
)
cq.exporters.export(result, 'GT.stl')