import cadquery as cq

result = (
cq.Workplane('front')
.circle(100)
.circle(120)
.workplane(offset=50)
.circle(100)
.circle(120)
.loft()
)
cq.exporters.export(result, 'GT.stl')