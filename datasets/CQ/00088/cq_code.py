import cadquery as cq

result = (
cq.Workplane('YZ')
.moveTo(-80.1, 23.36)
.lineTo(-13,33.36)
.hLineTo(-80.1)
.close()
.extrude(106, both=True)
)
cq.exporters.export(result, 'GT.stl')