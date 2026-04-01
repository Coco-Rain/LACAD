import cadquery as cq

result = (
cq.Workplane("YZ")
.rect(5,5)
.vertices("<Z")
.circle(2)
.extrude(-1)
cq.exporters.export(result, 'GT.stl')