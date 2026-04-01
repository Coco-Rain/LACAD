import cadquery as cq

result = (
cq.Workplane()
.lineTo(0,2)
.lineTo(4,3)
.lineTo(4,0)
.close()
.revolve(axisStart=(0,0,0), axisEnd=(1,0,0))
)
cq.exporters.export(result, 'GT.stl')